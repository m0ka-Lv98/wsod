import yaml
import json
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from utils import bbox_collate, MixedRandomSampler
import transform as transf
from dataset import MedicalBboxDataset

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
    dataset_means = json.load(open(config['dataset']['mean_file']))
    
    # train dataの取得
    transform = Compose([
        transf.Augmentation(config['augmentation']),
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])
    
    dataset_all = MedicalBboxDataset(
        config['dataset']['annotation_file'],
        config['dataset']['image_root'])
    if 'class_integration' in config['dataset']:
        dataset_all = dataset_all.integrate_classes(
            config['dataset']['class_integration']['new'],
            config['dataset']['class_integration']['map'])
    
    train_all = dataset_all.split(train, config['dataset']['split_file'])
    train_all.set_transform(transform)
    train_normal = train_all.without_annotation()
    train_anomaly = train_all.with_annotation()
    n_fg_class = len(dataset_all.get_category_names()) 

    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = MixedRandomSampler(
        [train_normal, train_anomaly],
        iteration * batchsize,
        ratio=[config['negative_ratio'], 1],
        generator=generator)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batchsize, drop_last=False)

    dataloader_train = DataLoader(
        sampler.get_concatenated_dataset(),
        num_workers=8,
        batch_sampler=batch_sampler,
        collate_fn=bbox_collate)
    #ここまで

    #test dataの取得
    transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])
    dataset_val = dataset_all.split(val, config['dataset']['split_file'])
    dataset_val.set_transform(transform)
    #ここまで

    return dataloader_train, dataset_val, dataset_all
