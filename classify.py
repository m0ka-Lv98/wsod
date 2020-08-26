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

from model import ResNet50, ResNet101
from efficientnet import efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3
from make_dloader import make_data
from utils import data2target, calc_confusion_matrix, draw_graph
from eval_classify import evaluate_coco_weak

torch.multiprocessing.set_sharing_strategy('file_system')
config = yaml.safe_load(open('./config.yaml'))

parser = argparse.ArgumentParser('classify')
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('--batchsize', type=int, default=config['batchsize'])
parser.add_argument('--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--tap', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
args = parser.parse_args()

metric_best = np.array([0,0,0])
metric_best_sum = 0
seed = time.time()


def main():
    dl_t, dl_v, _, _, _ = make_data(batchsize = args.batchsize, iteration = args.iteration, 
                                train = args.train, val = args.val)
    model = ResNet50(tap = args.tap)
    #model = efficientnet_b0()
    model.cuda()

    pos_weight = torch.tensor([18.0,18.0,18.0]*args.batchsize).reshape(-1,3)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)

    #訓練
    for epoch in range(args.epochs):
        train_val(model, optimizer, criterion, epoch, dl_t, dl_v)
        scheduler.step()
        

    """#モデルの最終評価
    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/wsod/model/resnet50_classify{val}.pt',
                        save_path = f"/data/unagi0/masaoka/wsod/result_bbox/resnet50_v{val}.json")
    """    
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
            torch.save(model.state_dict(), f'/data/unagi0/masaoka/wsod/model/cam/resnet50_cam{args.val}_{it}.pt')
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