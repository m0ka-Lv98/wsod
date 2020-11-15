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
from anchor import make_anchor

config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('-m', '--model', type=str, default="SLV_Retinaanchor")
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-p','--port',type=int,default=3289)
args = parser.parse_args()
seed = int(time.time()*100)
model_name = args.model+f'_{args.lr}'

def main():
    global model_name
    dataset_means = json.load(open(config['dataset']['mean_file']))
    oicr = resnet18(3,pretrained=True)
    if args.resume > 0:
        oicr.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{args.resume}.pt"))
    oicr = nn.DataParallel(oicr)
    oicr.cuda()
    torch.backends.cudnn.benchmark = True
    #opt = optim.Adam(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay)
    opt = optim.RMSprop(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    #if args.resume > 0:
        #opt.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{args.resume}.pt"))
        
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 2, gamma = 0.1)
    train_loss_list = []
    m_list = []
    l1_list = []
    l2_list = []
    l3_list = []
    ld_list = []
    loss_hist = collections.deque(maxlen=500)
    p_box = make_anchor()
    for epoch in range(args.resume,args.epochs):
        dl_t, dl_v, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val,p_path=0)
        for i, data in enumerate(dl_t,1):
            opt.zero_grad()
            labels, n, t, v, u= data2target(data)
            labels = labels.unsqueeze(1).unsqueeze(2).cuda().float() # bs, 1, 1, num_class
            rois = [p_box.cuda().float() for _ in range(labels.shape[0])]
            n = p_box.shape[0]
            rois = torch.stack(rois, dim=0) 
            rois = rois.unsqueeze(1) #bs, 1, n, 4
            output, loss,m,l1,l2,l3,lossd = oicr(data["img"].cuda().float(), labels, rois, n)
            ld = lossd/5
            loss = m+(l1+l2+l3)+ld
            loss = loss.mean()
            loss.backward()
            m = m.mean()
            l1 = l1.mean()
            l2 = l2.mean()
            l3 = l3.mean()
            ld = ld.mean()
            m_list.append(m.cpu().data.numpy())
            l1_list.append(l1.cpu().data.numpy())
            l2_list.append(l2.cpu().data.numpy())
            l3_list.append(l3.cpu().data.numpy())
            ld_list.append(ld.cpu().data.numpy())
            opt.step()
            train_loss_list.append(loss.cpu().data.numpy())
            if i%10==0:
                if i%100==0:
                    loss_hist.append(float(loss))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"{seed}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"{model_name}"))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(m_list)/len(m_list)]), 
                                win=f"{seed}", name='m', update='append',
                                opts=dict(showlegend=True))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(l1_list)/len(l1_list)]), 
                                win=f"{seed}", name='l1', update='append',
                                opts=dict(showlegend=True))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(l2_list)/len(l2_list)]), 
                                win=f"{seed}", name='l2', update='append',
                                opts=dict(showlegend=True))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(l3_list)/len(l3_list)]), 
                                win=f"{seed}", name='l3', update='append',
                                opts=dict(showlegend=True))
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(ld_list)/len(ld_list)]), 
                                win=f"{seed}", name='ld', update='append',
                                opts=dict(showlegend=True))
                train_loss_list = []
                m_list = []
                l1_list = []
                l2_list = []
                l3_list = []
                ld_list = []
            print(f'{i}/{len(dl_t)}, {loss}', end='\r')
        torch.save(oicr.module.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{epoch+1}.pt")
        torch.save(opt.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{epoch+1}.pt")
        #scheduler.step(np.mean(loss_hist))
        
        scheduler.step()
    


if __name__ == "__main__":
    viz = Visdom(port=args.port)
    main()