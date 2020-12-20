import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms
from utils.utils import *
import torchvision.models as models
import yaml
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F
import time
import numpy as np
import cv2
import math 
import torchvision
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from matplotlib import pyplot as plt
import c_utils
from numba import jit

config = yaml.safe_load(open('/home/mil/masaoka/wsod/config.yaml'))
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
dl_root = "/data/unagi0/masaoka/resnet_model_zoo/"

class MIDNFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7,mode='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = 0.25
        self.eps = eps
        self.mode=mode

    def forward(self, input, target):
        y = target #bs*proposal,4
        logit = input.clamp(self.eps, 1. - self.eps)
        loss = -(y * torch.log(logit)+(1-y)*torch.log(1-logit))# cross entropy
        alpha_factor = torch.ones(target.shape).cuda() * self.alpha
        alpha_factor = torch.where(torch.eq(target, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(target, 1.), 1. - logit, logit)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        loss = loss*focal_weight
        if self.mode=="mean":
            loss = (loss.sum(dim=1)).mean()

        return loss

class ICRFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7,mode='none'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.mode=mode

    def forward(self, input, target):
        y = target #bs*proposal,4
        
        logit = F.softmax(input,dim=-1) #bs*proposal,4
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -y * torch.log(logit) # cross entropy
        loss = loss * ((1 - logit) ** self.gamma) # focal loss
        if self.mode=="mean":
            loss = (loss.sum(dim=1)).mean()

        return loss

class ASPP(nn.Module):
    def  __init__(self,size_list=[]):
        super().__init__()
        assert len(size_list)>0
        self.avgpool_list = []
        for size in size_list:
            self.avgpool_list.append(nn.Sequential(nn.AdaptiveAvgPool2d(size),
                                                    nn.Flatten()))
        
    def forward(self,x):
        vec = []
        for avgpool in self.avgpool_list:
            vec.append(avgpool(x))
        vec = torch.cat(vec,1)
        return vec

class _ROIPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, rois):
        #rois: bs,2000,4 
        #output: bs, 2000, ch, h, w
        rois = [r for r in rois]
        h, w = inputs.shape[2], inputs.shape[3]
        res = roi_align(inputs, rois, 7, spatial_scale=w/512)
        return res

class vector_extractor(nn.Module):
    """input: images, proposals
         output: feature vector"""
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        out = 512
        layers = list(model.children())[:-2]
        self.feature_map = nn.Sequential(*layers)
        self.roi_pool = _ROIPool()
        #self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_vector = nn.Sequential(ASPP([1,2,3]),
                                            #nn.AdaptiveAvgPool2d(1),
                                            nn.Flatten(),
                                            nn.Linear(out*(14), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(out*(14), 3))
        
    def forward(self, inputs, rois):
        f = self.feature_map(inputs)
        roi_pool = self.roi_pool(f, rois)
        vec = self.feature_vector(roi_pool) 
      
        aux = self.aux(f)
        return vec,aux
      
    
class MIDN(nn.Module):
    """input: feature vector, labels
         output: scores per proposal, loss"""
    def __init__(self,c_in=512):
        super().__init__()
        self.c_in = c_in
        self.c_out = 4
        self.layer_c = nn.Linear(c_in, self.c_out)
        self.layer_d = nn.Linear(c_in, self.c_out)
        self.softmax_c = nn.Softmax(dim=2)
        self.softmax_d = nn.Softmax(dim=1)
        #self.loss = MIDNFocalLoss(gamma=2,mode='mean')#nn.BCELoss()
        self.loss = nn.BCELoss()
        self.aux_loss = nn.BCEWithLogitsLoss()
        self.upper = 1-1e-7
        self.lower = 1e-7
    def forward(self, inputs, labels, num, aux=None):
        bs, proposal = inputs.shape[0]//num, num
        x_c = self.layer_c(inputs).view(bs, proposal, -1) #bs, proposal, c_out
        x_d = self.layer_d(inputs).view(bs, proposal, -1)
        sigma_c = self.softmax_c(x_c)
        sigma_d = self.softmax_d(x_d)
        x_r = sigma_c * sigma_d #bs, proposal, c_out
        if not self.training:
            return x_r
        labels = labels[:,:self.c_out].clone().detach()
        phi_c = x_r.sum(dim=1) #bs, c_out
        phi_c = torch.clamp(phi_c, self.lower,self.upper)
        if isinstance(aux,torch.Tensor):
            loss = self.loss(phi_c, labels) + self.aux_loss(aux,labels)
        else:
            loss = self.loss(phi_c, labels)

       
        return x_r,loss
        

class ICR(nn.Module):
    """input: feature vector (bs*proposal, ch)
                     k-1th proposal scores(bs, proposal,3or4)
                     supervision (label) (bs)
                     ROI proposals
         output: refined proposal scores, loss"""
    def __init__(self,c_in=512):
        super().__init__()
        self.c_in = c_in
        self.I_t = 0.5
        self.fc = nn.Linear(c_in, 4)
        self.softmax = nn.Softmax(dim=2)
        self.fl = ICRFocalLoss(gamma=2)

    def forward(self, inputs, pre_score, labels, rois, num):
        bs, proposal = inputs.shape[0]//num, num
        pre_score = pre_score.clone().detach()
        labels = labels.clone().detach()
        xr_k = self.fc(inputs).view(bs, proposal, -1) #bs, proposal, 4
        logit = self.softmax(xr_k)
        if not self.training:
            return logit
        _xr_k = xr_k.view(bs*proposal, -1)
        
        y_k = np.zeros((bs,proposal,4)).astype('float32')
        y_k[:,:,3] = 1
        w = np.ones((bs,proposal)).astype('float32')
        I = -np.ones((bs,proposal)).astype('float32')
        for batch in range(bs):
            for c in range(3):
                j_list = torch.empty(0)
                if labels[batch][c]:
                    label = c
                    j_list = torch.nonzero(pre_score[batch,:,c]>0.5,as_tuple=False).squeeze()
                    x_list = pre_score[batch,j_list,c]
                    if (j_list).size() == torch.Size([0]) or (j_list).size() == torch.Size([1]) or (j_list).size() == torch.Size([]):
                        m = torch.max(pre_score[batch, :, c], 0)
                        x = m[0].item()
                        j = m[1].item()
                        x_list = torch.tensor([x])
                        j_list =torch.tensor([j])
                    
                    mat = box_iou(rois[batch], rois[batch][j_list])

                    mat = mat.cpu().detach().numpy()
                    j_list = j_list.cpu().detach().numpy()
                    x_list = x_list.cpu().detach().numpy()
                    y_k, w, I = c_utils.icr(mat,y_k,w,I,x_list,j_list,self.I_t,c,batch)
                        
        y_k = torch.from_numpy(y_k).cuda().float()
        w = torch.from_numpy(w).cuda().float()
        y_k = y_k.view(bs*proposal, -1)
        w = w.view(bs*proposal, 1)

        imbalance_list = y_k.float().sum(dim=0)
        loss = self.fl(_xr_k.float(), y_k)
        loss = (w*loss)/(imbalance_list+1e-7)#bs*proposal,4
        loss = loss.view(bs,proposal,-1)
        w_ = 10*torch.exp(logit)#bs,proposal,4
        lab = labels.unsqueeze(1)#bs,1,4
        lab[:,0,3] = 1
        mask = 1-lab
        w_ = w_*mask+lab
        loss = loss*w_
        loss = loss.sum()/bs

        '''loss = self.fl(_xr_k.float(), y_k)
        loss = loss*w
        loss = (loss/inputs.shape[0]).sum()'''

        
        return logit, loss

def generate_gt(scores_list,rois_list,labels):
    #scores:bs,proposal,4; rois_list:bs,n,4; labels:bs,4
    gt_list = []
    bs,proposal,_ = scores_list.shape
    for batch in range(bs):
        label = torch.max(labels[batch],0)[1]
        if label == 3:
            gt_list.append(torch.tensor([]))
            continue
        rois = rois_list[batch]
        scores = scores_list[batch]
        rois = rois[torch.max(scores,1)[1]!=3]
        scores = scores[torch.max(scores,1)[1]!=3]
        scores = scores[:,label]
        if len(scores)==0:
            gt_list.append(torch.tensor([]))
            continue
        index = nms(rois,scores,0.5)
        rois = rois[index]
        scores = scores[index]
        for i in range(len(scores)):
            if scores[i]>0.5:
                continue
            else:
                break
        rois = rois[:i]
        scores = scores[:i]
        gt_list.append(rois.clone().detach())
    return gt_list

class MultiScaleROIPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchvision.ops.MultiScaleRoIAlign(['feat0','feat1','feat2', 'feat3','feat4'], 7, 2)
    def forward(self,features,rois):
        i = OrderedDict()
        bs = rois.shape[0]
        image_size = [(512,512)]
        outputs = []
        for k in range(bs):
            i['feat0'] = features[0][k:k+1]
            i['feat1'] = features[1][k:k+1]
            i['feat2'] = features[2][k:k+1]
            i['feat3'] = features[3][k:k+1]
            i['feat4'] = features[4][k:k+1]
            r = rois[k]
            
            outputs.append(self.m(i, [r], image_size))
        print((torch.cat(outputs,0)).shape)
        return torch.cat(outputs,0)

class fmap_extractor(nn.Module):
    """input: images, proposals
         output: feature vector"""
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(model.children())[:4])
        self.layer1 = list(model.children())[4]
        self.layer2 = list(model.children())[5]
        self.layer3 = list(model.children())[6]
        self.layer4 = list(model.children())[7]
        
    def forward(self, inputs):
        x = self.base(inputs)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4

class Detection(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(512,4)
        self.reg = nn.Linear(512,4)
        self.reg.weight.data.fill_(0)
        self.reg.bias.data.fill_(0)
        prior = 0.01
        self.cls.weight.data.fill_(0)
        self.cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.l1 = nn.SmoothL1Loss(reduction='none')
        self.zero = nn.Parameter(torch.tensor(0.)).requires_grad_(False)
    def forward(self,v,gt_list,rois_list,labels,pre_score):
        if not self.training:
            cls = F.softmax(self.cls(v),dim=-1)#proposal,4
            reg = self.reg(v)#proposal,4
            tx,ty,tw,th = reg[:,0],reg[:,1],reg[:,2],reg[:,3]
            r = rois_list[0]
            rx = (r[:,2]+r[:,0])/2
            ry = (r[:,3]+r[:,1])/2
            rw = (r[:,2]-r[:,0])/2
            rh = (r[:,3]-r[:,1])/2
            gx = tx*rw+rx
            gy = ty*rh+ry
            gw = rw*torch.exp(tw)
            gh = rh*torch.exp(th)
            x0 = gx-gw/2
            x1 = gx+gw/2
            y0 = gy-gh/2
            y1 = gy+gh/2
            boxes = torch.stack([x0,y0,x1,y1],1)
            return boxes, cls
        bs = len(gt_list)
        pre_score = pre_score.clone().detach()
        classify = self.cls(v).view(bs,-1,4) #bs,proposal,4
        proposal = classify.shape[1]
        reg = self.reg(v).view(bs,-1,4)
        
        c_list = 0
        c_list_bg = 0
        l1_list= 0
        i = 0
        bg = 0
        for batch in range(bs):
            label = torch.max(labels[batch],0)[1]
            c = torch.clamp(classify[batch],1e-7,1-1e-7)
            if gt_list[batch].size()==torch.Size([0,4]) or gt_list[batch].size()==torch.Size([0]):
                bg+=proposal
                target = torch.tensor([label]*proposal).cuda()
                c_loss = self.ce(classify[batch],target).mean()
                if c_list_bg==0:
                    c_list_bg = c_loss
                else:
                    c_list_bg+=c_loss
                continue
            gt_list[batch] = gt_list[batch].float()
            i+=1
            target = torch.tensor([label]*proposal).cuda()
            c_loss = self.ce(c,target)*pre_score[batch,:,label] #proposal,4
            ious = box_iou(gt_list[batch],rois_list[batch])
            _ious = ious>0.5
            mask = _ious.any(dim=0)
            c_loss = (c_loss*mask).sum()/(mask.sum()+1e-7)
            index = torch.max(ious,0)[1]
            g = gt_list[batch][index]
            gx = (g[:,2]+g[:,0])/2
            gy = (g[:,3]+g[:,1])/2
            gw = (g[:,2]-g[:,0])/2
            gh = (g[:,3]-g[:,1])/2
            r = rois_list[batch]
            rx = (r[:,2]+r[:,0])/2
            ry = (r[:,3]+r[:,1])/2
            rw = (r[:,2]-r[:,0])/2
            rh = (r[:,3]-r[:,1])/2
            tx = (gx-rx)/(rw+1e-8)
            ty = (gy-ry)/(rh+1e-8)
            tw = torch.log(gw/(rw+1e-8))
            th = torch.log(gh/(rh+1e-8))
            t = torch.stack([tx,ty,tw,th],1)
            l1_loss = (self.l1(reg[batch],t).sum(dim=1)*mask*pre_score[batch,:,label]).sum()/(bs*proposal)#mask.sum()
            if c_list==0:
                c_list = c_loss
            else:
                c_list+=c_loss
            if l1_list==0:
                l1_list = l1_loss
            else:
                l1_list+=l1_loss

        return (c_list+c_list_bg/(bg+1e-7))/(bs+1e-7)+l1_list

def generate_gt_retina(scores_list,rois_list,labels):
    #scores:bs,proposal,4; rois_list:bs,n,4; labels:bs,4
    gt_list = []
    bs,proposal,_ = scores_list.shape
    for batch in range(bs):
        label = torch.max(labels[batch],0)[1]
        if label == 3:
            gt_list.append(torch.empty(0,5))
            continue
        rois = rois_list[batch]
        scores = scores_list[batch]
        rois = rois[torch.max(scores,1)[1]!=3]
        scores = scores[torch.max(scores,1)[1]!=3]
        scores = scores[:,label]
        if len(scores)==0:
            gt_list.append(torch.empty(0,5))
            continue
        index = nms(rois,scores,0.5)
        rois = rois[index]
        scores = scores[index]
        for i in range(len(scores)):
            if scores[i]>0.5:
                continue
            else:
                break
        rois = rois[:i]
        l = torch.ones((rois.shape[0],1)).cuda()*label
        rois = torch.cat([rois,l.cuda()],dim=1)
        print(rois,rois.shape)
        gt_list.append(rois.clone().detach())
    return gt_list

@jit
def compute_mat(mat, rois, score):
    for r in range(len(rois)):
        mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            
    return mat 

