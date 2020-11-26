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
from dataset.make_dloader import make_data

config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('-m', '--model', type=str, default="SLV")
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-p','--port',type=int,default=3289)
args = parser.parse_args()
seed = int(time.time())
model_name = args.model+f'{args.lr}'
dl_t, dl_v, _, _, _ = make_data(batchsize=args.batchsize,iteration=args.iteration,val=args.val)
def main():
    dataset_means = json.load(open(config['dataset']['mean_file']))
    oicr = eval(args.model+'()')
    if args.resume > 0:
        oicr.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_{args.resume}.pt"))
    oicr = nn.DataParallel(oicr)
    oicr.cuda()
    torch.backends.cudnn.benchmark = True
    opt = optim.Adam(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay)
    #opt = optim.RMSprop(oicr.parameters(), lr = args.lr, weight_decay=args.weightdecay,momentum=0.9)
    if args.resume > 0:
        opt.load_state_dict(torch.load(f"/data/unagi0/masaoka/wsod/model/oicr/{model_name}_opt{args.resume}.pt"))
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 1, gamma = 0.1)
    train_loss_list = []
    m_list = []
    l1_list = []
    l2_list = []
    l3_list = []
    ld_list = []
    for epoch in range(args.resume,args.epochs):
        for i, data in enumerate(dl_t,1):
            opt.zero_grad()
            labels, n, t, v, u= data2target(data)
            labels = labels.unsqueeze(1).unsqueeze(2).cuda().float() # bs, 1, 1, num_class
            rois = [r.cuda().float() for r in data["p_bboxes"]]
            n = min(list(map(lambda x: x.shape[0], rois)))
            n = min(n,2000)
            for ind, tensor in enumerate(rois):
                rois[ind] = rois[ind][:n,:]
            rois = torch.stack(rois, dim=0) 
            rois = rois.unsqueeze(1) #bs, 1, n, 4
            output, loss,m,l1,l2,l3,ld = oicr(data["img"].cuda().float(), labels, rois, n)
            loss = m+l1+l2+l3+ld#*min(1,epoch+i/len(dl_t))
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
        #scheduler.step()
    


if __name__ == "__main__":
    viz = Visdom(port=args.port)
    main()