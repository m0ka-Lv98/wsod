import torch

from modules.models import *
from dataset.faster_dloader import make_data
from eval import coco_eval ,multi_eval, oicr_eval

if __name__=='__main__':
    val=0
    name = 'fasterwoicrfocalminiweight'
    model = faster_rcnn18(num_classes=3, pretrained=True)
    model.cuda()
    model.load_state_dict(torch.load(f'/data/unagi0/masaoka/wsod/model/oicr/F-RCNNwoicrfocalminiweight1e-05_0_6.pt'))
    dataset = torch.load(f'/data/unagi0/masaoka/val_all{val}.pt')
    coco_eval.evaluate_coco(dataset, model, val, name, threshold=0.05)

#fasterwoicr
#fasterwoicrfocal
#fasterwoicrfocalmini
#fasterwoicrfocalminiweight