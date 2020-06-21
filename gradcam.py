import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from make_dloader import make_data
from torch.utils.data import DataLoader
from utils import bbox_collate, MixedRandomSampler,count_parameters_in_MB
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
from losses import *


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return cam
   
def thresh(narray, threshold = 0.15, binary = False):
    if binary:
        return np.where(narray>threshold*np.max(narray), 1, 0)
    return np.where(narray>threshold*np.max(narray), narray, 0)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        o = torch.sigmoid(output)
        o = o.cpu().data.numpy()
        if index == None:
            o = np.where(o>0.5, 1., 0.)
        label = o.sum(axis = 0)
        label = np.where(label>o.shape[0]/2, 1., 0.)
        cam_list = []
        for i in range(len(label)):
            if label[i] == 0:
                cam_list.append(0)
                continue  
            one_hot = np.zeros_like(label)
            one_hot[i] = 1.
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)
            
            self.feature_module.zero_grad()
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy() # 40,2048,16,16

            weights = np.mean(grads_val, axis=(2, 3)) #40,2048
            weights = weights[:,:,np.newaxis,np.newaxis] #40,2048,1,1
            cam = np.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32) #40,16,16
            target =  weights * target #40,2048,16,16
            target = target.sum(axis=1) #40,16,16
            target = np.maximum(target, 0)
            T = np.zeros((input.shape[0],input.shape[2],input.shape[3]))
            for b in range(input.shape[0]):
                cam = cv2.resize(target[b], input.shape[2:])
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                T[b] = cam
            cam_list.append(T)
            
        if input.shape[0] == 1:
            return cam_list
        return label, cam_list

def converge_map(init, mask):
    #init 512,512 ndarray ; mask b,512,512 ndarray
    seq1 = iaa.Sequential([
                    iaa.Affine(
        rotate=iap.DiscreteUniform(-180,170)*(-1)
                    )])
    ia.seed(0)
    mask_reconvert = torch.from_numpy(seq1(images=mask)) #b,512,512 tensor
    mp = torch.from_numpy(init).requires_grad_(True) #512,512 tensor
    #optimizer = optim.SGD([mp], lr = 1e-3)
    #optimizer = optim.LBFGS([mp])
    calc_loss = Loss()
    calc_loss.cuda()
    optimizer = optim.Adam([mp], lr = 5e-3)
    losses = []
    #for j in range(mask.shape[0]):
    #    plt.imshow(mask_reconvert[j])
    #    plt.show()
    for i in range(200):
        optimizer.zero_grad()
        loss = calc_loss(mp.cuda(), mask_reconvert.cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss)
    #plt.figure()
    #plt.plot(range(len(losses)), losses, linestyle = '-', color = 'red', label = 'loss')
    #plt.xlabel('times')
    #plt.ylabel('loss')
    #plt.legend()
    #plt.show()
    mp = mp.data.numpy()
    return mp #512,512 ndarray


def augmented_grad_cam(gcam, img):           #gcamはheatmapとlabelを出力するクラス
    #img.shape=B,C,H,W　tensor
    img = img.squeeze().numpy().transpose(1,2,0)  #512,512,3
    b = 20
    img = np.tile(img,(b,1,1,1)) #b,512,512,3
    seq = iaa.Sequential([
                    iaa.Affine(
        rotate=iap.DiscreteUniform(-180, 179)
                    )])
    ia.seed(0)
    img = seq(images=img)
    img = torch.from_numpy(img)
    labels, masks = gcam(img.permute(0,3,1,2), None) #label [1,0,1] masks [(40,512,512), 0, (40,512,512)]
    maps = []
    for i, label in enumerate(labels):
        if label == 0:
            maps.append(0)
            continue
        else:
            init_map = masks[i][0] #512,512
            mp = converge_map(init_map, masks[i])
            maps.append(mp[np.newaxis,:,:])
    
    return maps

def heatmap2box(heatmap, img=0, threshold = 0.5):
    # img 512,512,3 ndarray,      heatmap  1,512,512 ndarray
    if not isinstance(img, numbers.Number):
        image = img.copy()
    heatmap = thresh(heatmap, threshold = threshold)
    heatmap = heatmap[0]
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

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

