import torch.nn as nn
import torch
import math
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numbers
from cam import heatmap2box, CAM

class ResNet50(CAM):
    def __init__(self):
        super(CAM,self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_class = 3 #torose, vascular, ulcer
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=self.num_class, bias=True))



    