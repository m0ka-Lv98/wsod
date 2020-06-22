import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from make_dloader import make_data
import json
import yaml
import numpy as np
from model import ResNet50
import yaml
from visdom import Visdom
from utils import bbox_collate
from eval_classify import evaluate_coco_weak
import time

config = yaml.safe_load(open('./config.yaml'))
batchsize = 100
iteration = 2000
epochs = 10
option = 0
seed = time.time()


def main():
    if option == 0:
        model = ResNet50()
    elif option == 1:
        model = Unet()
    model.cuda()

    pos_weight = torch.ones(batchsize, 3)
    for i in range(pos_weight.size(0)):
        pos_weight[i][0] = 1.5
        pos_weight[i][1] = 8.0
        pos_weight[i][2] = 8.0
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())

    for i in range(pos_weight.size(0)):
        pos_weight[i][0] = 7.5
        pos_weight[i][1] = 64.0
        pos_weight[i][2] = 86.5
    criterion_val = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        train(model, optimizer, criterion, criterion_val, pos_weight, epoch)
    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/resnet50_classify{val}.pt',
                        save_path = f"/data/unagi0/masaoka/resnet50_v{val}.json", aug = False)
    
def train(model, optimizer, criterion, criterion_val, pos_weight, epoch):
    global metric_best
    model.train()
    train_loss_list = []
    for it, data in enumerate(dataloader_train, 1):
        optimizer.zero_grad()
        output = model(data['img'].cuda().float())
        target, n,t,v,u = data2target(data, output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        output = torch.sigmoid(output).cpu().data.numpy()
        output = np.where(output>0.5,1,0)
        target = target.cpu().data.numpy().astype(np.uint8)
        result = (output - target)==0
        acc = result.all(axis=1).sum()/output.shape[0]
        train_loss_list.append(loss.cpu().data.numpy())
        if it%10==0:
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{seed}_{val}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title="Loss"))
            del train_loss_list
            train_loss_list = []
        if (it+epoch*iteration)==1 or it%500==0:
            tp,fp,fn,tn = valid(model, dataloader_val, criterion_val, pos_weight)
            recall = tp/(tp+fn)
            specificity = tn/(fp+tn)
            metric = recall + specificity - 1
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[0]]), \
                                win=f'metric{seed}_{val}', name='torose', update='append',
                                opts=dict(showlegend=True,title=f"Recall+Specifity-1 val{val}"))
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[1]]), \
                                win=f'metric{seed}_{val}', name='vascular', update='append',
                                opts=dict(showlegend=True))
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric[2]]), \
                                win=f'metric{seed}_{val}', name='ulcer', update='append',
                                opts=dict(showlegend=True))
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([metric.mean()]), \
                                win=f'metric{seed}_{val}', name='average', update='append',
                                opts=dict(showlegend=True))
            if metric.sum() > metric_best:
                torch.save(model.state_dict(), f'/data/unagi0/masaoka/resnet50_classify{val}.pt')
                metric_best = metric.sum()
        print(f'{epoch}epoch,{it}/{len(dataloader_train)}, loss {loss.data:.4f}, acc {acc:.3f}')

def valid(model, d_val, criterion_val, pos_weight):
    tpa , fpa, fna, tna = np.zeros(3), np.zeros(3), np.zeros(3), 0
    with torch.no_grad():
        for i, d in enumerate(d_val):
            print(f"validating {i}/{len(d_val)}", end = '\r')
            scores = torch.sigmoid(model(d['img'].cuda().float()))
            output = scores.cpu().data.numpy()
            output = np.where(output>0.5,1,0)
            target, n, t, v, u= data2target(d, torch.from_numpy(output))
            target = target.cpu().data.numpy()
            gt = np.array([n,t,v,u])
            tp, fp, fn, tn = calc(output, target, gt)
            tpa += tp
            fpa += fp
            fna += fn
            tna += tn
    return tpa, fpa, fna, tna

def calc(output, target, gt):
    tp = (output * target).sum(axis = 0)
    fp = (output * (1 - target)).sum(axis = 0)
    fn = gt[1:] - tp
    tn = np.all((1 - output) * (1 - target), axis = 1).sum()
    return tp, fp, fn, tn


#入力データ(バッチ)から教師データに変換　変換後：[[0,0,1],[0,0,0],...]
def data2target(data, output):
    target = torch.zeros_like(output)
    n = 0
    t = 0
    v = 0
    u = 0
    for i in range(output.shape[0]):
        bbox = data["annot"][i][:,:]
        bbox = bbox[bbox[:,4]!=-1]
        flag = -1
        for k in range(bbox.shape[0]):
            flag = int(bbox[k][4])
            target[i][flag] = 1
            n+=1
        if flag == 0:
            t+=1
        elif flag == 1:
            v+=1
        elif flag == 2:
            u+=1
    target.cuda()
    return target,n,t,v,u

if __name__ == "__main__":
    viz = Visdom(port=3289)
    val = config['dataset']['val'][0]
    metric_best = 0
    dataloader_train, dataset_val, _ = make_data(batchsize=batchsize, iteration = iteration)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=40, shuffle=False, 
                                                            num_workers=4, collate_fn=bbox_collate)
    main()