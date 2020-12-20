import yaml
import json
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from utils.utils import bbox_collate, MixedRandomSampler
import utils.transform as transf
from .dataset import MedicalBboxDataset
import time

def make_data(batchsize = None, iteration = None, train = None, val = None, p_path='str'):
    config = yaml.safe_load(open('/home/mil/masaoka/wsod/config.yaml'))
    if batchsize == None: 
        batchsize = config["batchsize"] 
    if iteration == None:
        iteration = config["n_iteration"]
    if train == None:
        train = config['dataset']['train']
    if val == None:
        val = config['dataset']['val']
    else:
        val = [val]
    dataset_means = json.load(open(config['dataset']['mean_file']))

    try:
        train_normal = torch.load(f'/data/unagi0/masaoka/train_normal_{val[0]}.pt')
        print('loaded normal')
        train_torose = torch.load(f'/data/unagi0/masaoka/train_torose_{val[0]}.pt')
        print('loaded torose')
        train_vascular = torch.load(f'/data/unagi0/masaoka/train_vascular_{val[0]}.pt')
        print('loaded vascular')
        train_ulcer = torch.load(f'/data/unagi0/masaoka/train_ulcer_{val[0]}.pt')
        print('loaded ulcer')
        #train_all = torch.load(f'/data/unagi0/masaoka/train_all_{val[0]}.pt')
        #train_ulcer = train_all.ulcer()
        #torch.save(train_ulcer,f'/data/unagi0/masaoka/train_ulcer_{val[0]}.pt')
    
    except:
        dataset_all = MedicalBboxDataset(
            config['dataset']['annotation_file'],
            config['dataset']['image_root'],
            pseudo_path='0')
        # train dataの取得
        transform = Compose([
            transf.Augmentation(config['augmentation']),
            transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
            transf.Normalize(dataset_means['mean'], dataset_means['std']),
            transf.HWCToCHW()
            ])
    
        if 'class_integration' in config['dataset']:
            dataset_all = dataset_all.integrate_classes(
                config['dataset']['class_integration']['new'],
                config['dataset']['class_integration']['map'])
    
        train_all = dataset_all.split(train, config['dataset']['split_file'])
        train_all.set_transform(transform)
        torch.save(train_all,f'/data/unagi0/masaoka/train_all_{val[0]}.pt')
        print('save all!')
        train_normal = train_all.without_annotation()
        torch.save(train_normal,f'/data/unagi0/masaoka/train_normal_{val[0]}.pt')
        print('save normal!')
        train_torose = train_all.torose()
        torch.save(train_torose,f'/data/unagi0/masaoka/train_torose_{val[0]}.pt')
        print('save torose!')
        train_vascular = train_all.vascular()
        torch.save(train_vascular,f'/data/unagi0/masaoka/train_vascular_{val[0]}.pt')
        print('save vascular!')
        train_ulcer = train_all.ulcer()
        torch.save(train_ulcer,f'/data/unagi0/masaoka/train_ulcer_{val[0]}.pt')
        print('save ulcer!')
    

    generator = torch.Generator()
    generator.manual_seed(int(time.time()*10000)%1000000007)
    #train_anomaly, train_normal
    #train_normal, train_anomaly
    sampler = MixedRandomSampler(
        [train_normal,train_torose,train_vascular,train_ulcer],
        iteration * batchsize,
        ratio=[20,20,20,20],
        #ratio=[len(train_normal),len(train_torose),len(train_vascular),len(train_ulcer)],
        generator=generator)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batchsize, drop_last=False)
    
    dataloader_train = DataLoader(
        sampler.get_concatenated_dataset(),
        num_workers=4,
        batch_sampler=batch_sampler,
        collate_fn=bbox_collate)
    #ここまで

    #test dataの取得
    transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])

    val_all=0
    dataloader_val=0
    
    try:
        val_all = torch.load(f'/data/unagi0/masaoka/val_all{val[0]}.pt')
    except:
        val_all = dataset_all.split(val, config['dataset']['split_file'])
        val_all.set_transform(transform)
        torch.save(val_all,f'/data/unagi0/masaoka/val_all{val[0]}.pt') 
    dataloader_val = DataLoader(val_all, batch_size=1, shuffle=False, 
                                num_workers=4, collate_fn=bbox_collate)
    #ここまで'''
    return dataloader_train, 0, dataloader_val, val_all, 0
