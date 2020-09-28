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
    def __init__(self, model, clamp=False):
        super().__init__()
        self.model = model
        self.extractor = model.extractor #modelで定義しておく
        w = model.fc_w()   #modelで定義しておく
        if clamp:
            self.w = torch.relu(w)
        else:
            self.w = w
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



