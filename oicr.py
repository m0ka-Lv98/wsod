import torch
import torch.nn as nn
import torch.optim as optim
from modules import OICR
import torchvision
from torchvision import models
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose
import transform as transf
from torch.utils.data import DataLoader
from utils import bbox_collate, data2target,MixedRandomSampler
import yaml
import os
import json
import copy
from PIL import Image
from dataset import MedicalBboxDataset
from make_dloader import make_data
from visdom import Visdom
import argparse

config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('-m', '--model', type=str, default="ResNet18")
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-p','--port',type=int,default=3289)
args = parser.parse_args()

model_name = args.model+f'scaledfocal0.5{args.lr}'
dl_t, dl_v, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val)
def main():
    dataset_means = json.load(open(config['dataset']['mean_file']))
    oicr = OICR()
    if args.resume > 0:
        oicr.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{args.resume}.pt"))
    oicr = nn.DataParallel(oicr)
    oicr.cuda()
    torch.backends.cudnn.benchmark = True
    #opt = optim.Adam(oicr.parameters(), lr = 1e-4, weight_decay=1e-5)
    opt = optim.RMSprop(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    if args.resume > 0:
        opt.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{args.resume}.pt"))
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 1, gamma = 0.1)
    train_loss_list = []
    for epoch in range(args.resume,args.epochs):
        for i, data in enumerate(dl_t):
            opt.zero_grad()
            labels, n, t, v, u= data2target(data)
            rois = [r.cuda().float() for r in data["p_bboxes"]]
            n = min(list(map(lambda x: x.shape[0], rois)))
            for ind, tensor in enumerate(rois):
                rois[ind] = rois[ind][:n,:]
            n = min(2000,n)
            rois = torch.stack(rois, dim=0)[:,:n,:] #bs, n, 4
            rois = rois.unsqueeze(1) #bs, 1, n, 4
            output, loss = oicr(data["img"].cuda().float(),labels.unsqueeze(1).unsqueeze(2).cuda().float(),rois, n)
            loss = loss.mean()
            loss.backward()
            opt.step()
            train_loss_list.append(loss.cpu().data.numpy())
            if i%10==0:
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{0}_{0}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"Loss_val{0}"))
                del train_loss_list
                train_loss_list = []
            print(f'{i}/{len(dl_t)}, {loss}', end='\r')
        torch.save(oicr.module.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{epoch+1}.pt")
        torch.save(opt.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{epoch+1}.pt")
        scheduler.step()
    


if __name__ == "__main__":
    viz = Visdom(port=args.port)
    main()