import torch.nn as nn
import torch
import math
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numbers
from cam import heatmap2box
from modules import *
from efficientnet_pytorch import EfficientNet


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, tap=False):
        super().__init__()
        model = models.resnet50(pretrained=pretrained)
        layers = list(model.children())[:-2] #特徴マップまで
        self.layers = nn.Sequential(*layers)
        self.head = 1
        self.num_classes = num_classes #torose, vascular, ulcer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=self.num_classes, bias=False))
    def forward(self, x):
        feature = self.extractor(x)
        feature = self.gap(feature).squeeze(2).squeeze(2)
        output = self.fc(feature)
        return output

    def fc_w(self):
        return self.fc.weight

    def extractor(self, x):
        feature = self.layers(x)
        return feature

class DualResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, head=1, tap=False):
        super().__init__()
        self.num_classes = num_classes #torose, vascular, ulcer
        self.tap = tap
        model = models.resnet50(pretrained=pretrained)
        layers = list(model.children())[:-2] #特徴マップまで
        self.layers = nn.Sequential(*layers)
        self.head = head
        self.sa = SA(2048, head)
        self.ca = CA(2048)
        self.s = nn.Parameter(torch.tensor([0.]))
        self.c = nn.Parameter(torch.tensor([0.]))
        self.gap = nn.AdaptiveAvgPool2d(1) if not tap else TAP()
        self.fc = nn.Linear(4096, self.num_classes, bias=False)

    def forward(self, x):
        x = self.extractor(x)
        x = self.gap(x)
        x = x.view(-1, 4096)
        x = self.fc(x)
        return x

    def fc_w(self):
        return self.fc.weight

    def extractor(self, x):
        feature = self.layers(x)
        o = self.sa(feature)
        r = self.ca(feature)
        s_feature = feature*o
        c_feature = feature*r
        cat_feature = torch.cat([s_feature, c_feature], dim=1)
        return cat_feature
        #return c_feature
    

class ResNet101(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, tap=False):
        super().__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self.num_classes = num_classes #torose, vascular, ulcer
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=self.num_classes, bias=False))

    def forward(self, x):
        output = self.model(x)
        return output

    def fc_w(self):
        for n, p in self.model.named_parameters():
            if n == 'fc.0.weight':
                return p
    def extractor(self, x):
        layers = list(self.model.children())[:-2]
        m = nn.Sequential(*layers)
        feature = m(x)
        return feature

class EfficientNetb0(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Sequential(nn.Linear(in_features=1280, out_features=self.num_classes, bias=False))

    def forward(self, x):
        feature = self.model.extract_features(x)
        feature = self.gap(feature)
        feature = feature.squeeze(3).squeeze(2)
        out = self.model.fc(feature)
        return out

    def fc_w(self):
        for n, p in self.model.named_parameters():
            if n == "fc.0.weight":
                return p

    def extractor(self, x):
        feature = self.model.extract_features(x)
        return feature
    

class EfficientNetb1(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Sequential(nn.Linear(in_features=1280, out_features=self.num_classes, bias=False))

    def forward(self, x):
        feature = self.model.extract_features(x)
        feature = self.gap(feature)
        feature = feature.squeeze(3).squeeze(2)
        out = self.model.fc(feature)
        return out

    def fc_w(self):
        for n, p in self.model.named_parameters():
            if n == "fc.0.weight":
                return p

    def extractor(self, x):
        feature = self.model.extract_features(x)
        return feature
