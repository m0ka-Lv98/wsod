import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from make_dloader import make_data
import json
import yaml
import numpy as np
from model import ResNet50, EfficientNetb0, EfficientNetb1
from efficientnet import efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3
import yaml
from visdom import Visdom
from utils import bbox_collate, data2target, calc_confusion_matrix, draw_graph
from eval_classify import evaluate_coco_weak
import time
import optuna
import sys 
import os 
import argparse
import logging

torch.multiprocessing.set_sharing_strategy('file_system')
config = yaml.safe_load(open('./config.yaml'))
parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs="*", type=int, default=config['dataset']['train'])
parser.add_argument('--val', type=int, default=config['dataset']['val'][0])
parser.add_argument('--batchsize', type=int, default=config['batchsize'])
parser.add_argument('--iteration', type=int, default=config['n_iteration'])
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--tap', action='store_true', default=False)
parser.add_argument('--trialsize', type=int, default=20)
parser.add_argument('--model', type=str, default="ResNet50")
args = parser.parse_args()

model_opt = 0
step = 1
seed = time.time()
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format)
fh = logging.FileHandler(os.path.join('optuna', f'{args.model}.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    logging.info(f'args:{args}')
    TRIAL_SIZE = args.trialsize
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    for trial in study.get_trials():
        logging.info(f'{trial.number}: {trial.value:.3f} ({trial.params})')
    
    logging.info(f'best value: {study.best_value}')
    logging.info(f'best params: {study.best_params}')
    
def get_optimizer(trial, model):
    adam_lr = trial.suggest_loguniform('lr', 1e-7, 1e-3)
    weight_decay = trial.suggest_loguniform('weightdecay', 1e-11, 1e-8)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    return optimizer

def objective(trial):
    global step
    print(f'{step}/{args.trialsize}')
    step += 1
    best_score = 1
    EPOCHS=args.epochs
    model = eval(args.model + "()")
    model = nn.DataParallel(model)
    model.tap = args.tap
    model.cuda()

    t = trial.suggest_discrete_uniform("p_w_t", 5.0, 15.0, 1.0)
    v = trial.suggest_discrete_uniform("p_w_v", 5.0, 15.0, 1.0)
    u = trial.suggest_discrete_uniform("p_w_u", 1.0, 15.0, 1.0)
    optimizer = get_optimizer(trial, model)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)
    pos_weight = torch.tensor([t,u,v]*args.batchsize).reshape(-1,3)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight.cuda().float())
    for step in range(EPOCHS):
        metric = -train_val(model, optimizer, criterion, step, dataloader_train, dataloader_val)
        scheduler.step()
        if metric < best_score:
            best_score = metric
    return best_score

def train_val(model, optimizer, criterion, epoch, d_train, d_val):
    metric_best = -1
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
            viz.line(X = np.array([it + epoch*args.iteration]),Y = np.array([sum(train_loss_list)/len(train_loss_list)]), 
                                win=f"t_loss{seed}_{args.val}", name='train_loss', update='append',
                                opts=dict(showlegend=True,title=f"Loss_val{args.val}"))
            del train_loss_list
            train_loss_list = []
        if it%500==0:
            model.eval()
            tp,fp,fn,tn = valid(model, d_val)
            precision = tp/(tp+fp+1e-10)
            recall = tp/(tp+fn+1e-10)
            specificity = tn/(fp+tn+1e-10)
            metric = 2*recall*precision/(recall+precision+1e-10)
            draw_graph(precision, recall, specificity, metric, seed, args.val, epoch, args.iteration, it, viz)
            if metric.sum()/3 > metric_best:
                #torch.save(model.state_dict(), f'/data/unagi0/masaoka/resnet50_classify{args.val}.pt')
                metric_best = metric.sum()/3
            model.train()
        
    return metric_best

def valid(model, d_val):
    model.eval()
    tpa , fpa, fna, tna = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    with torch.no_grad():
        for i, d in enumerate(d_val):
            print(f"validate {i}/{len(d_val)}", end = '\r')
            scores = torch.sigmoid(model(d['img'].cuda().float()))
            output = scores.cpu().data.numpy()
            output = np.where(output>0.5,1,0)
            target, n, t, v, u= data2target(d, torch.from_numpy(output))
            target = target.cpu().data.numpy()
            gt = np.array([n,t,v,u])
            tp, fp, fn, tn = calc_confusion_matrix(output, target, gt)
            del target
            del output
            del d
            tpa += tp
            fpa += fp
            fna += fn
            tna += tn
            
    return tpa, fpa, fna, tna
    
    
if __name__ == "__main__":
    viz = Visdom(port=3289)
    optuna.logging.disable_default_handler()
    dataloader_train, dataloader_val, _, _, _ = make_data(batchsize=args.batchsize, iteration=args.iteration)
    main()