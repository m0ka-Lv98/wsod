import torch.nn as nn
import torch
import math
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numbers
from cam import heatmap2box
from efficientnet_pytorch import EfficientNet

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, tap=False):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
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
