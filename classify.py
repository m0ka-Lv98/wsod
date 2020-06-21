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

config = yaml.safe_load(open('./config.yaml'))
batchsize = 100
iteration = 2500
epochs = 10
option = 0

def main():
    if option == 0:
        model = ResNet50()
    elif option == 1:
        model = Unet()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        train(model, optimizer, epoch)
    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/resnet50_classify{val}.pt',
                        save_path = f"/data/unagi0/masaoka/resnet50_v{val}.json", aug = False)
    
def train(model, optimizer, epoch):
    global loss_best
    model.train()
    train_loss_list = []
    for it, data in enumerate(dataloader_train, 1):
        optimizer.zero_grad()
        output = model(data['img'].cuda().float())
        target, weight = data2target(data, output)
        loss = BCELoss(output, target, weight)
        loss.backward()
        optimizer.step()
        output = torch.sigmoid(output).cpu().data.numpy()
        output = np.where(output>0.5,1,0)
        target = target.cpu().data.numpy().astype(np.uint8)
        result = (output - target)==0
        acc = result.all(axis=1).sum()/output.shape[0]
        train_loss_list.append(loss.cpu().data.numpy())
        if it%500==0:
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=20, shuffle=False, 
                                                            num_workers=4, collate_fn=bbox_collate)
            loss_val = valid(model, dataloader_val)
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), \
                                win='loss', name='train_loss', update='append')
            viz.line(X = np.array([it + epoch*iteration]),Y = np.array([loss_val]), \
                                win='loss', name='val_loss', update='append')
            del train_loss_list
            train_loss_list = []
            if loss_best > loss_val:
                torch.save(model.state_dict(), f'/data/unagi0/masaoka/resnet50_classify{val}.pt')
                loss_best = loss_val
        print(f'{epoch}epoch,{it}/{len(dataloader_train)}, loss {loss.data:.4f}, acc {acc:.3f}')

def valid(model, dataloader_val):
    loss = 0
    with torch.no_grad():
        for i, d in enumerate(dataloader_val):
            print(f"validating {i}/{len(dataloader_val)}", end = '\r')
            output = model(d['img'].cuda().float())
            target, weight = data2target(d, output)
            loss += BCELoss(output, target, weight)/len(dataset_val)
    return loss.cpu().data.numpy()


#入力データ(バッチ)から教師データに変換　変換後：[[0,0,1],[0,0,0],...]
#lossに対するweightも出力
def data2target(data, output):
    target = torch.zeros_like(output)
    weight = torch.ones_like(output)
    for i in range(output.shape[0]):
        bbox = data["annot"][i][:,:]
        bbox = bbox[bbox[:,4]!=-1]
        flag = -1
        for k in range(bbox.shape[0]):
            flag = int(bbox[k][4])
            target[i][flag] = 1
        if flag == 0:
            w = 7.5
        elif flag == 1:
            w = 64
        elif flag == 2:
            w = 86.5
        if flag != -1:
            weight[i][flag] = w
    target.cuda()
    return target, weight

def BCELoss(output, target, weight):
    loss = target * (torch.log(torch.sigmoid(output) + 1e-10)) + \
           (1 - target) * (torch.log(1 - torch.sigmoid(output) + 1e-10))
    loss = -(loss * weight).sum()/(output.shape[0])
    return loss

if __name__ == "__main__":
    viz = Visdom(port=3289)
    val = config['dataset']['val'][0]
    loss_best = 100
    dataloader_train, dataset_val, _ = make_data(batchsize=batchsize, iteration = iteration)
    main()