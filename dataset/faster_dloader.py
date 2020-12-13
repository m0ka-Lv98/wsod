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
    
    dataset_all = MedicalBboxDataset(
        config['dataset']['annotation_file'],
        config['dataset']['image_root'],
        pseudo_path=0)
        # train dataの取得
    transform = Compose([
        transf.Augmentation(config['augmentation']),
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()])
    
    if 'class_integration' in config['dataset']:
        dataset_all = dataset_all.integrate_classes(
            config['dataset']['class_integration']['new'],
            config['dataset']['class_integration']['map'])
    
    train_all = dataset_all.split(train, config['dataset']['split_file'])
    train_all.set_transform(transform)
    train_normal = train_all.without_annotation()
    train_torose = train_all.torose()
    train_vascular = train_all.vascular()
    train_ulcer = train_all.ulcer() 

    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = MixedRandomSampler(
        [train_normal,train_torose,train_vascular,train_ulcer],
        iteration * batchsize,
        ratio=[20,20,20,1],
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
    val_all = dataset_all.split(val, config['dataset']['split_file'])
    val_all.set_transform(transform)
    
    dataloader_val = DataLoader(val_all, batch_size=1, shuffle=False, 
                                num_workers=4, collate_fn=bbox_collate)
    #ここまで'''
    return dataloader_train, train_all, dataloader_val, val_all, 1
