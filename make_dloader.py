import yaml
import json
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from utils import bbox_collate, MixedRandomSampler
import transform as transf
from dataset import MedicalBboxDataset
import time
def make_data(batchsize = None, iteration = None, train = None, val = None):
    config = yaml.safe_load(open('./config.yaml'))
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
    p_path = "/data/unagi0/masaoka/endoscopy/annotations/pseudo_annotations500.json"
    dataset_means = json.load(open(config['dataset']['mean_file']))
    
    try:
        #print('all')
        print('ulcer')
        train_ulcer = torch.load('/data/unagi0/masaoka/train_ulcer.pt')
        #train_all = torch.load('/data/unagi0/masaoka/train_all.pt')
        print('normal')
        train_normal = train_ulcer#torch.load('/data/unagi0/masaoka/train_normal.pt')#
        print('torose')
        train_torose = train_normal#torch.load('/data/unagi0/masaoka/train_torose.pt')#
        print('vascular')
        train_vascular = train_normal#torch.load('/data/unagi0/masaoka/train_vascular.pt')#
        
    except:
        dataset_all = MedicalBboxDataset(
            config['dataset']['annotation_file'],
            config['dataset']['image_root'],
            pseudo_path=p_path)
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
        torch.save(train_all,'/data/unagi0/masaoka/train_all.pt')
        print('save complete!')
        train_normal = train_all.without_annotation()
        train_normal.set_transform(transform)
        torch.save(train_normal,'/data/unagi0/masaoka/train_normal.pt')
        train_torose = train_all.torose()
        train_torose.set_transform(transform)
        torch.save(train_torose,'/data/unagi0/masaoka/train_torose.pt')
        train_vascular = train_all.vascular()
        train_vascular.set_transform(transform)
        torch.save(train_vascular,'/data/unagi0/masaoka/train_vascular.pt')
        train_ulcer = train_all.ulcer()
        train_ulcer.set_transform(transform)
        torch.save(train_ulcer,'/data/unagi0/masaoka/train_ulcer.pt')
    #n_fg_class = len(dataset_all.get_category_names()) 

    generator = torch.Generator()
    generator.manual_seed(0)
    #train_anomaly, train_normal
    #train_normal, train_anomaly
    sampler = MixedRandomSampler(
        [train_normal,train_torose,train_vascular,train_ulcer],
        iteration * batchsize,
        ratio=[10,10,10,1],
        generator=generator)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batchsize, drop_last=False)
    
    dataloader_train = DataLoader(
        sampler.get_concatenated_dataset(),
        num_workers=8,
        batch_sampler=batch_sampler,
        collate_fn=bbox_collate)
    #ここまで

    '''#test dataの取得
    transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])

    try:
        s= time.time()
        val_all = torch.load(f'/data/unagi0/masaoka/val_all{val[0]}.pt')
        e=time.time()
        print(e-s)
    except:
        val_all = dataset_all.split(val, config['dataset']['split_file'])
        val_all.set_transform(transform)
        torch.save(val_all,f'/data/unagi0/masaoka/val_all{val[0]}.pt')
    dataloader_val = DataLoader(val_all, batch_size=1, shuffle=False, 
                                num_workers=4, collate_fn=bbox_collate)
    #ここまで'''
    del train_normal,train_torose,train_ulcer,train_vascular
    return dataloader_train, 1, 1, 1, 1
