import torch

from modules.models import *
from dataset.fast_dloader import make_data
from eval import fast_eval

if __name__=='__main__':
    val=0
    name = 'fast'
    model = fast_rcnn18(num_classes=3, pretrained=True)
    model.cuda()
    model.load_state_dict(torch.load(f'/data/unagi0/masaoka/wsod/model/oicr/Fast-RCNNanchor1e-05_0_6.pt'))
    dataset = torch.load(f'/data/unagi0/masaoka/val_all{val}.pt')
    fast_eval.evaluate_coco(dataset, model, val, name, threshold=0.05)
