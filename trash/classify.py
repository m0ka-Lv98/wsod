import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import json
import yaml
import numpy as np
import yaml
from visdom import Visdom
import time
import argparse

from model import *
from make_dloader import make_data
from utils import data2target, calc_confusion_matrix, draw_graph
from eval_classify import evaluate_coco_weak

torch.multiprocessing.set_sharing_strategy('file_system')
config = yaml.safe_load(open('./config.yaml'))

parser = argparse.ArgumentParser('classify')
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('-b','--batchsize', type=int, default=config['batchsize'])
parser.add_argument('-i', '--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weightdecay', type=float, default=1e-7)
parser.add_argument('-m', '--model', type=str, default="DualResNet50")
parser.add_argument('-np', '--not_pretrained', action='store_false', default=True)
parser.add_argument('--tap', action="store_true", default=False)
args = parser.parse_args()

metric_best = np.array([0,0,0])
metric_best_sum = 0
seed = time.time()
Dir = "tap" if args.tap else "cam"
model_name = args.model
if args.not_pretrained == False:
    model_name += '_not_pretrained'


def main():
    global model_name
    dl_t, dl_v, _, _, _ = make_data(batchsize = args.batchsize, iteration = args.iteration, 
                                train = args.train, val = args.val)
    model = eval(args.model + f'(pretrained={args.not_pretrained}, tap={args.tap})')
    if model.head != 1:
        model_name += f'{model.head}head'
    model = torch.nn.DataParallel(model) # make parallel
    model.cuda()
    torch.backends.cudnn.benchmark = True
    

    pos_weight = torch.tensor([9.,16.0,13.]*args.batchsize).reshape(-1,3)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weightdecay, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weightdecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)

    #訓練
    for epoch in range(args.epochs):
        train_val(model, optimizer, criterion, epoch, dl_t, dl_v)
        #scheduler.step()
    
    torch.save(model.module.state_dict(), f'/data/unagi0/masaoka/wsod/model/{Dir}/{model_name}_{args.val}.pt')
        
    #モデルの最終評価
    evaluate_coco_weak(args.val, model = args.model, tap = model.tap, model_path = f'/data/unagi0/masaoka/wsod/model/{Dir}/{model_name}_{args.val}.pt',
                        save_path = f"/data/unagi0/masaoka/wsod/result_bbox/{Dir}/{model_name}_{args.val}.json")
       
def train_val(model, optimizer, criterion, epoch, dl_t, dl_v):
    global metric_best_sum
    global metric_best
    model.train()
    train_loss_list = []
    for it, data in enumerate(dl_t, 1):
        optimizer.zero_grad()
        output = model(data['img'].cuda().float())
        target, n,t,v,u = data2target(data, output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.cpu().data.numpy())
        print(f'{epoch}epoch,{it}/{len(dl_t)}, loss {loss.data:.4f}', end='\r')
        if it%10==0:
            viz.line(X = np.array([it + epoch*args.iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{seed}_{args.val}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"Loss_val{args.val}"))
            del train_loss_list
            train_loss_list = []
        if it%500==0:
            tp,fp,fn,tn = valid(model, dl_v)
            precision = tp/(tp+fp+1e-10)
            recall = tp/(tp+fn)
            specificity = tn/(fp+tn)
            metric = 2*recall*precision/(recall+precision+1e-10)
            draw_graph(precision, recall, specificity, metric, seed, args.val, epoch, args.iteration, it, viz)
            torch.save(model.module.state_dict(), f'/data/unagi0/masaoka/wsod/model/{Dir}/{model_name}_{args.val}_{epoch*args.iteration+it}.pt')
            metric_best = metric if metric.sum() > metric_best.sum() else metric_best
            metric_best_sum = metric_best.sum()
            model.train()

def valid(model, dl_v):
    model.eval()
    tpa , fpa, fna, tna = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    with torch.no_grad():
        for i, d in enumerate(dl_v):
            print(f'validation {i}/{len(dl_v)}', end='\r')
            scores = torch.sigmoid(model(d['img'].cuda().float()))
            output = scores.cpu().data.numpy()
            output = np.where(output>0.5,1,0)
            target, n, t, v, u = data2target(d, torch.from_numpy(output))
            target = target.cpu().data.numpy()
            gt = np.array([n,t,v,u])
            tp, fp, fn, tn = calc_confusion_matrix(output, target, gt)
            tpa += tp
            fpa += fp
            fna += fn
            tna += tn
    return tpa, fpa, fna, tna
    
    
if __name__ == "__main__":
    viz = Visdom(port=3289)
    main()