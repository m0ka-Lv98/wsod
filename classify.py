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

from model import ResNet50
from make_dloader import make_data
from utils import bbox_collate, data2target, calc_confusion_matrix, draw_graph
from eval_classify import evaluate_coco_weak

torch.multiprocessing.set_sharing_strategy('file_system')
config = yaml.safe_load(open('./config.yaml'))
val = config['dataset']['val'][0]
batchsize = config["batchsize"]
iteration = config["n_iteration"]
epochs = 2
metric_best = 0
seed = time.time()


def main():
    dataloader_train, dataset_val, _ = make_data()
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=40, shuffle=False, 
                                                num_workers=4, collate_fn=bbox_collate)
    model = ResNet50()
    model.cuda()

    pos_weight = torch.tensor([18.0,18.0,18.0]*batchsize).reshape(-1,3)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda())
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    #訓練
    for epoch in range(epochs):
        train_val(model, optimizer, criterion, epoch, dataloader_val)

    #モデルの最終評価
    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/wsod/model/resnet50_classify{val}.pt',
                        save_path = f"/data/unagi0/masaoka/wsod/result_bbox/resnet50_v{val}.json")
    
def train_val(model, optimizer, criterion, epoch, d_val):
    global metric_best
    dataloader_train, _, _ = make_data()
    model.train()
    train_loss_list = []
    for it, data in enumerate(dataloader_train, 1):
        optimizer.zero_grad()
        output = model(data['img'].cuda().float())
        target, n,t,v,u = data2target(data, output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.cpu().data.numpy())
        print(f'{epoch}epoch,{it}/{len(dataloader_train)}, loss {loss.data:.4f}', end='\r')
        if it%10==0:
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{seed}_{val}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"Loss_val{val}"))
            del train_loss_list
            train_loss_list = []
        if (it+epoch*iteration)==1 or it%500==0:
            tp,fp,fn,tn = valid(model, d_val)
            precision = tp/(tp+fp+1e-10)
            recall = tp/(tp+fn)
            specifity = tn/(fp+tn)
            metric = 2*recall*precision/(recall+precision+1e-10)
            draw_graph(recall, specifity, metric, seed, val, epoch, iteration, it, viz)
            if metric.sum() > metric_best:
                torch.save(model.state_dict(), f'/data/unagi0/masaoka/wsod/model/resnet50_classify{val}.pt')
                metric_best = metric.sum()
            model.train()

def valid(model, d_val):
    model.eval()
    tpa , fpa, fna, tna = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    with torch.no_grad():
        for i, d in enumerate(d_val):
            print(f'validation {i}/{len(d_val)}', end='\r')
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