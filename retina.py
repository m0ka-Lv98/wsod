import torch
import torch.nn as nn
import torch.optim as optim
from modules import *
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
import time
import collections

config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('-m', '--model', type=str, default="Retina")
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-p','--port',type=int,default=3289)
args = parser.parse_args()
seed = int(time.time())
model_name = args.model+f'{args.lr}'
dl_t, dl_v, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val)
def main():
    global model_name
    dataset_means = json.load(open(config['dataset']['mean_file']))
    oicr = resnet18_full(3,pretrained=True)
    oicr.pre.load_state_dict(torch.load("/data/unagi0/masaoka/wsod/model/oicr/SLV_Retinarms0.1sched1e-05_4.pt"))#SLV_Retina_prepre1e-05_2.pt
    #oicr.load_state_dict(torch.load("/data/unagi0/masaoka/wsod/model/oicr/SLV_Retina_prepre1e-05_6.pt"),strict=False)
    if args.resume > 0:
        oicr.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{args.resume}.pt"))
    oicr = nn.DataParallel(oicr)
    oicr.cuda()
    torch.backends.cudnn.benchmark = True
    opt = optim.Adam(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay)
    #opt = optim.RMSprop(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    if args.resume > 0:
        opt.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{args.resume}.pt"))
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True)
    train_loss_list = []
    loss_hist = collections.deque(maxlen=500)
    for epoch in range(args.resume,args.epochs):
        for i, data in enumerate(dl_t,1):
            opt.zero_grad()
            closs,rloss = oicr(data["img"].cuda().float(), data['p_bboxes'])
            loss = closs.mean()+rloss.mean()
            loss.backward()
            
            opt.step()
            train_loss_list.append(loss.cpu().data.numpy())
            if i%10==0:
                if i%100==0:
                    loss_hist.append(float(loss))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"{seed}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"{model_name}"))
                
                train_loss_list = []
                
            print(f'{i}/{len(dl_t)}, {loss}', end='\r')
        torch.save(oicr.module.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{epoch+1}.pt")
        torch.save(opt.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{epoch+1}.pt")
        scheduler.step(np.mean(loss_hist))
    


if __name__ == "__main__":
    viz = Visdom(port=args.port)
    main()

#Retinafully0.0001 教師ありretinanet