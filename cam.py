import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from make_dloader import make_data
from torch.utils.data import DataLoader
from utils import bbox_collate, MixedRandomSampler
import transform as transf
import yaml
import json
from matplotlib import pyplot as plt
import numbers
import torchvision
import torch.optim as optim
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import copy


def thresh(narray, threshold = 0.15, binary = False):
    if binary:
        return np.where(narray>threshold*np.max(narray), 1, 0)
    return np.where(narray>threshold*np.max(narray), narray, 0)


def heatmap2box(heatmap, img=0, threshold = 0.5):
    # img 512,512,3 ndarray,      heatmap  512,512 ndarray
    if not isinstance(img, numbers.Number):
        image = img.copy()
    heatmap = thresh(heatmap, threshold = threshold)
    heatmap = np.uint8(255*heatmap)
    label = cv2.connectedComponentsWithStats(heatmap)
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    boxes = torch.tensor([])
    score = 0
    for i in range(n):
    # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        if boxes.shape[0] == 0:
            boxes = torch.tensor([[x0,y0,x1,y1]])
        else:
            boxes = torch.cat((boxes, torch.tensor([[x0,y0,x1,y1]])), dim=0)
        score = threshold
        if not isinstance(img, numbers.Number):
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 0), thickness = 4)
    if not isinstance(img, numbers.Number):
        plt.imshow(image)
        plt.show()
    return boxes, score

class Cam(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.extractor = model.extractor #modelで定義しておく
        self.w = model.fc_w()   #modelで定義しておく
        self.model.eval()
    def forward(self, input):
        shape = input.shape[2:]
        fmap = self.extractor(input)
        fmap = fmap.data.cpu().numpy()[0]
        w = self.w.data.cpu().numpy()
        output = self.model(input)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        output = np.where(output > 0.5, 1, 0)[0]
        maps = []
        w = w[:, :, np.newaxis, np.newaxis]
        for c_id in range(self.model.num_classes):
            if output[c_id] == 0:
                maps.append(0)
                continue
            temp = fmap*w[c_id] #2048,16,16
            temp = temp.sum(axis=0)
            temp = cv2.resize(temp, shape)
            temp = np.where(temp > 0, temp, 0)
            temp = temp/temp.max()
            maps.append(temp)
        return output, maps

class Gen_bbox:
    def __init__(self, cam):
        self.cam = cam
    
    def __call__(self, input):
        #input 1,3,512,512
        _, masks = self.cam(input)
        boxes = torch.tensor([])
        scores = torch.tensor([])
        classification = torch.tensor([])
        for threshold in reversed(range(11)):
            threshold = threshold/10
            for num, mask in enumerate(masks):
                if isinstance(mask,numbers.Number):
                    continue
                #mask 512,512
                #print(threshold)
                box, score = heatmap2box(mask, threshold = threshold)
                #print(box)
                if box.shape[0] == 0:
                    continue
                if boxes.shape[0] == 0:
                    boxes = box
                else:
                    boxes = torch.cat((boxes,box), dim = 0)
                if scores.shape[0] == 0:
                    scores = torch.Tensor(len(box))
                    scores.fill_(score)

                else:
                    s = torch.Tensor(len(box))
                    s.fill_(score)
                    scores = torch.cat((scores, s))
                if classification.shape[0] == 0:
                    classification = torch.Tensor(len(box))
                    classification.fill_(num)
                else:
                    c = torch.Tensor(len(box))
                    c.fill_(num)
                    classification = torch.cat((classification, c))

        return scores, classification.int(), boxes



"""
class CAM(nn.Module):
    def __init__(self, tap = False):
        super().__init__()
        self.num_class = 3 #torose, vascular, ulcer
        self.model = None
        self.tap = tap
        self.theta = 0.05

    def forward(self,x):
        if not self.tap:
            x = self.model(x)
            return x
        for name, module in self.model._modules.items():
            if name == "avgpool" and self.tap:
                f_max = x.view(x.shape[0],x.shape[1],-1).max(axis = 2)[0]
                f_max = f_max.view(f_max.shape[0], -1, 1, 1)
                f = x/f_max
                f_ones = torch.where(f > self.theta, torch.tensor([1.0]).cuda(), torch.tensor([0.0]).cuda())
                x = (x*f_ones).view(f_ones.shape[0], f_ones.shape[1], -1).sum(axis = 2).unsqueeze(2).unsqueeze(3)
                x = x/((f_ones.view(f_ones.shape[0], f_ones.shape[1], -1).sum(axis = 2) + 1e-7).unsqueeze(2).unsqueeze(3)) #元論文
                #x = x/((x + 1e-7).unsqueeze(2).unsqueeze(3)) #こっちの方がいいのでは？
                #x = module(x)
                x = x.view(x.shape[0], x.shape[1])
            else:
                x = module(x)
        return x
    
    def cam(self, x, nwc = False):
        #x 1,3,512,512
        self.model.eval()
        shape = x.shape[2:] #input shape 512,512
        output = self.model(x)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        output = np.where(output > 0.5, 1, 0)[0]
        for name, module in self.model._modules.items():
            if name == "avgpool":
                break
            x = module(x)
        maps = []
        for params in self.model.fc.parameters():
            weights = params #3,2048
            break
        x = x[0].data.cpu().numpy() #2048,16,16
        w = weights.data.cpu().numpy() #3,2048

        if nwc:
            w = np.where(w > 0, w, 0)

        w = w[:, :, np.newaxis, np.newaxis]
        for c_id in range(self.num_class):
            if output[c_id] == 0:
                maps.append(0)
                continue
            temp = x*w[c_id] #2048,16,16
            temp = temp.sum(axis=0)
            temp = cv2.resize(temp, shape)
            temp = np.where(temp > 0, temp, 0)
            temp = temp/temp.max()
            
            maps.append(temp)
        return output, maps

    def gen_bbox(self, x):
        #x 1,3,512,512
        _, masks = self.cam(x)
        boxes = torch.tensor([])
        scores = torch.tensor([])
        classification = torch.tensor([])
        for threshold in reversed(range(11)):
            threshold = threshold/10
            for num, mask in enumerate(masks):
                if isinstance(mask,numbers.Number):
                    continue
                #mask 512,512
                #print(threshold)
                box, score = heatmap2box(mask, threshold = threshold)
                #print(box)
                if box.shape[0] == 0:
                    continue
                if boxes.shape[0] == 0:
                    boxes = box
                else:
                    boxes = torch.cat((boxes,box), dim = 0)
                if scores.shape[0] == 0:
                    scores = torch.Tensor(len(box))
                    scores.fill_(score)

                else:
                    s = torch.Tensor(len(box))
                    s.fill_(score)
                    scores = torch.cat((scores, s))
                if classification.shape[0] == 0:
                    classification = torch.Tensor(len(box))
                    classification.fill_(num)
                else:
                    c = torch.Tensor(len(box))
                    c.fill_(num)
                    classification = torch.cat((classification, c))

        return scores, classification.int(), boxes
"""


