import os
import json
import yaml
from visdom import Visdom
import argparse
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modules.models import *
from utils.anchor import make_anchor
from utils.utils import data2target
from dataset.fast_dloader import make_data
import subprocess


config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('-e', '--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('-m', '--model', type=str, default="Fast-RCNN")
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-p','--port',type=int,default=3289)
parser.add_argument('-g','--gpu',type=str,default='0')
args = parser.parse_args()
seed = int(time.time()*100)
model_name = args.model+f'{args.lr}_{args.val}'

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def main():
    global model_name
    dataset_means = json.load(open(config['dataset']['mean_file']))
    oicr = fast_rcnn18(3,pretrained=True)
    if args.resume > 0:
        oicr.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{args.resume}.pt"))
    oicr = nn.DataParallel(oicr)
    oicr.cuda()
    torch.backends.cudnn.benchmark = True
    #opt = optim.Adam(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay)
    #opt = optim.SGD(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    opt = optim.RMSprop(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    if args.resume > 0:
        opt.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{args.resume}.pt"))
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=1, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 1, gamma = 0.5)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3, eta_min=0.001)
    train_loss_list = []
    m_list = []
    l1_list = []
    l2_list = []
    l3_list = []
    lf_list = []
    loss_hist = collections.deque(maxlen=500)
    box = make_anchor()
    #dl_t, _, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val)
    for epoch in range(args.resume,args.epochs):
        dl_t, _, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val)
        for i, data in enumerate(dl_t,1):
            opt.zero_grad()
            labels, n, t, v, u= data2target(data)
            labels = labels.cuda().float() # bs, num_class
            p_list = []
            for r in data["p_bboxes"]:
                if r.shape==torch.Size([0]):
                    p_list.append(box)
                else:
                    p_list.append(r)
            

            rois = [r.cuda().float() for r in p_list]
            n = min(list(map(lambda x: x.shape[0], rois)))
            n = min(n,2000)
            print(n)
            for ind, tensor in enumerate(rois):
                rois[ind] = rois[ind][:n,:]
            rois = torch.stack(rois, dim=0)

            output, loss,m,l1,l2,l3,lf = oicr(data["img"].cuda().float(), labels, rois)
            lf = lf*min(1,epoch+(i/len(dl_t)))
            loss = m+l1+l2+l3+lf
            loss = loss.mean()
            loss.backward()
            
            m = m.mean()
            l1 = l1.mean()
            l2 = l2.mean()
            l3 = l3.mean()
            lf = lf.mean()
            m_list.append(m.cpu().data.numpy())
            l1_list.append(l1.cpu().data.numpy())
            l2_list.append(l2.cpu().data.numpy())
            l3_list.append(l3.cpu().data.numpy())
            lf_list.append(lf.cpu().data.numpy())
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
                viz.line(X = np.array([i + epoch*len(dl_t)]),Y = np.array([sum(lf_list)/len(lf_list)]), 
                                win=f"{seed}", name='lf', update='append',
                                opts=dict(showlegend=True))
                train_loss_list = []
                m_list = []
                l1_list = []
                l2_list = []
                l3_list = []
                lf_list = []
            print(f'{i}/{len(dl_t)}, {loss}', end='\r')
        torch.save(oicr.module.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{epoch+1}.pt")
        torch.save(opt.state_dict(), f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{epoch+1}.pt")
        #scheduler.step(np.mean(loss_hist))
        #if epoch>0:
        #    scheduler.step()
        scheduler.step()
        #if epoch ==1:
        #    opt = optim.RMSprop(oicr.parameters(), lr = 1e-4, weight_decay=args.weightdecay,momentum=0.9)


if __name__ == "__main__":
    viz = Visdom(port=args.port)
    main()

#/data/unagi0/masaoka/wsod/model/oicr/F-RCNNo51e-05_0_6.pt roi pooling後fc層でfaster rcnn. c = self.RCNN_c(v)