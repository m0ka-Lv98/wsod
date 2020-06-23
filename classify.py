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
from utils import bbox_collate, data2target, calc_confusion_matrix, draw_graph
from eval_classify import evaluate_coco_weak
import time

config = yaml.safe_load(open('./config.yaml'))
val = config['dataset']['val'][0]
batchsize = 100
iteration = 2000
epochs = 10
model_opt = 0
metric_best = 0
seed = time.time()


def main():
    dataloader_train, dataset_val, _ = make_data(batchsize=batchsize, iteration = iteration)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=40, shuffle=False, 
                                                num_workers=4, collate_fn=bbox_collate)
    if model_opt == 0:
        model = ResNet50()
    elif model_opt == 1:
        model = Unet()
    
    model.load_state_dict(torch.load(f"/data/unagi0/masaoka/resnet50_classify4.pt"))
    model.cuda()


    pos_weight = torch.tensor([1.5,8.0,8.0]*batchsize).reshape(-1,3)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        train_val(model, optimizer, criterion, epoch, dataloader_train, dataloader_val)

    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/resnet50_classify{val}.pt',
                        save_path = f"/data/unagi0/masaoka/resnet50_v{val}.json", aug = False)
    
def train_val(model, optimizer, criterion, epoch, d_train, d_val):
    global metric_best
    model.train()
    train_loss_list = []
    for it, data in enumerate(d_train, 1):
        optimizer.zero_grad()
        output = model(data['img'].cuda().float())
        target, n,t,v,u = data2target(data, output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.cpu().data.numpy())
        print(f'{epoch}epoch,{it}/{len(d_train)}, loss {loss.data:.4f}', end='\r')
        if it%10==0:
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{seed}_{val}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"Loss_val{val}"))
            del train_loss_list
            train_loss_list = []
        if (it+epoch*iteration)==1 or it%500==0:
            tp,fp,fn,tn = valid(model, d_val)
            recall = tp/(tp+fn)
            specifity = tn/(fp+tn)
            print(tp,fp,fn,tn)
            metric = recall + specifity - 1
            draw_graph(recall, specifity, seed, val, epoch, iteration, it, viz)
            if metric.sum() > metric_best:
                torch.save(model.state_dict(), f'/data/unagi0/masaoka/resnet50_classify{val}.pt')
                metric_best = metric.sum()
            model.train()

def valid(model, d_val):
    #model = ResNet50()
    #model.load_state_dict(torch.load(f"/data/unagi0/masaoka/resnet50_classify4.pt"))
    model.eval()
    tpa , fpa, fna, tna = np.zeros(3), np.zeros(3), np.zeros(3), 0
    with torch.no_grad():
        for i, d in enumerate(d_val):
            print(f"validate {i}/{len(d_val)}")
            scores = torch.sigmoid(model(d['img'].cuda().float()))
            output = scores.cpu().data.numpy()
            output = np.where(output>0.5,1,0)
            #print(scores)
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
    """dataloader_train, dataset_val, _ = make_data(batchsize=batchsize, iteration = iteration)
    d_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, 
                                                num_workers=4, collate_fn=bbox_collate)
    model = ResNet50()
    model.cuda()
    model.load_state_dict(torch.load(f"/data/unagi0/masaoka/resnet50_classify0.pt"))
    valid(model,d_val)"""
    main()