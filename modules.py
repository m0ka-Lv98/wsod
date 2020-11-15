import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms
from utils import *
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

config = yaml.safe_load(open('./config.yaml'))
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
dl_root = "/data/unagi0/masaoka/resnet_model_zoo/"

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7,mode='none'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.mode=mode

    def forward(self, input, target):
        y = target #bs*proposal,4
        
        logit = F.softmax(input,dim=-1) #bs*proposal,4
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit) # cross entropy
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
    def __init__(self):
        super().__init__()
        c_in = 512
        self.layer_c = nn.Linear(c_in, 3)
        self.layer_d = nn.Linear(c_in, 3)
        self.softmax_c = nn.Softmax(dim=2)
        self.softmax_d = nn.Softmax(dim=1)
        self.loss = nn.BCELoss()
        self.aux_loss = nn.BCEWithLogitsLoss()
        self.upper = 1-1e-7
        self.lower = 1e-7
    def forward(self, inputs, labels, num, aux):
        labels = labels[:,:-1].clone().detach()
        bs, proposal = inputs.shape[0]//num, num
        x_c = self.layer_c(inputs).view(bs, proposal, -1) #bs, proposal, 3
        x_d = self.layer_d(inputs).view(bs, proposal, -1)
        sigma_c = self.softmax_c(x_c)
        sigma_d = self.softmax_d(x_d)
        x_r = sigma_c * sigma_d #bs, proposal, 3
        if not self.training:
            return x_r
        phi_c = x_r.sum(dim=1) #bs, 3
        phi_c = torch.clamp(phi_c, self.lower,self.upper)
        loss = self.loss(phi_c, labels) + self.aux_loss(aux,labels)
       
        return  x_r,loss
        

class ICR(nn.Module):
    """input: feature vector (bs*proposal, ch)
                     k-1th proposal scores(bs, proposal,3or4)
                     supervision (label) (bs)
                     ROI proposals
         output: refined proposal scores, loss"""
    def __init__(self):
        super().__init__()
        c_in = 512
        self.I_t = 0.5
        self.fc = nn.Linear(c_in, 4)
        self.softmax = nn.Softmax(dim=2)
        self.fl = FocalLoss(gamma=2)
        self.one = nn.Parameter(torch.tensor([1.])).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        
    def forward(self, inputs, pre_score, labels, rois, num):
        bs, proposal = inputs.shape[0]//num, num
        pre_score = pre_score.clone().detach()
        labels = labels.clone().detach()
        xr_k = self.fc(inputs).view(bs, proposal, -1) #bs, proposal, 4
        logit = self.softmax(xr_k)
        if not self.training:
            return logit
        _xr_k = xr_k.view(bs*proposal, -1)
        y_k = torch.zeros(bs, proposal, 4).cuda()
        y_k[:, :, 3] = 1
        
        w = torch.stack([torch.cat([self.one]*proposal)]*bs,0)
        I = torch.stack([torch.cat([-self.one]*proposal)]*bs,0)
        for batch in range(bs):
            for c in range(3):
                if labels[batch][c]:
                    label = c
                    j_list = torch.nonzero(pre_score[batch,:,c]>0.5).squeeze()
                    x_list = pre_score[batch,j_list,c]
                    if (j_list).size() == torch.Size([0]) or (j_list).size() == torch.Size([1]) or (j_list).size() == torch.Size([]):
                        m = torch.max(pre_score[batch, :, c], 0)
                        x = m[0].item()
                        j = m[1].item()
                        x_list = [x]
                        j_list =[j]
                    mat = box_iou(rois[batch], rois[batch][j_list])
                    for i,j in enumerate(j_list):
                        _I = mat[:,i]
                        old = I[batch].clone()
                        I[batch] = torch.where(_I > old,_I,old)
                        pre = w[batch].clone()
                        w[batch] = torch.where(_I > old,self.one*x_list[i],pre)
                        p = y_k[batch, :, 3].clone()
                        q = y_k[batch, :, c].clone()
                        y_k[batch, :, 3] = torch.where(_I > self.I_t,self.zero,p)
                        y_k[batch, :, c] = torch.where(_I > self.I_t,self.one,q)

                        
                
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
        loss = loss.mean()'''

        
        return logit, loss

class OICR(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_extractor = vector_extractor()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        
    def forward(self, inputs, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            v,aux = self.v_extractor(inputs, rois)
            x, midn_loss = self.midn(v, labels, num, aux)
            x, loss1 = self.icr1(v, x, labels, rois, num)
            x, loss2 = self.icr2(v, x, labels, rois, num)
            x, loss3 = self.icr3(v, x, labels, rois, num) 
            loss = midn_loss + (loss1 + loss2 + loss3)
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0)
        else:
            rois = rois.squeeze().unsqueeze(0)
            v,_ = self.v_extractor(inputs, rois)
            #x = self.midn(v, labels, num)
            x1 = self.icr1(v, v, v, rois, num) 
            x2 = self.icr2(v, v, v, rois, num) 
            x3 = self.icr3(v, v, v, rois, num) 
            x = (x1+x2+x3)/3
            x, rois = x[0], rois[0].cuda()
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.5)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes

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
        self.m = torchvision.ops.MultiScaleRoIAlign(['feat2', 'feat3','feat4'], 7, 2)
    def forward(self,x2,x3,x4,rois):
        i = OrderedDict()
        i['feat2'] = x2
        i['feat3'] = x3
        i['feat4'] = x4
        image_size = [(512,512)]
        rois = [r for r in rois]
        output = self.m(i, rois, image_size)
        return output

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
            
                
                
class OICRe2e(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmap_extractor = fmap_extractor()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        self.detection = Detection()
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(512*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        
    def forward(self, inputs, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            f2,f3,f4 = self.fmap_extractor(inputs)
            f2 = torch.cat([f2]*4,1)
            f3 = torch.cat([f3]*2,1)
            f = self.multi_scale_roi_pool(f2, f3, f4, rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num)
            x, loss1 = self.icr1(v, x, labels, rois, num)
            x, loss2 = self.icr2(v, x, labels, rois, num)
            x, loss3 = self.icr3(v, x, labels, rois, num) 
            gt_list = generate_gt(x,rois,labels)            
            loss_detection = self.detection(v,gt_list,rois,labels,x)
            loss = midn_loss + (loss1 + loss2 + loss3)+loss_detection
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),loss_detection.unsqueeze(0)
        else:
            #rois = rois.squeeze()
            v = self.v_extractor(inputs, rois)
            rois,x = self.detection(v,None,rois,None,None)
            
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.5)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes


class SLV(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmap_extractor = fmap_extractor()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        self.detection = Detection()
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(512*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        
    def forward(self, inputs, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            f2,f3,f4 = self.fmap_extractor(inputs)
            f2 = torch.cat([f2]*4,1)
            f3 = torch.cat([f3]*2,1)
            f = self.multi_scale_roi_pool(f2, f3, f4, rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num) 
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels)          
            loss_detection = self.detection(v,gt_list,rois,labels,x)
            loss = midn_loss + (loss1 + loss2 + loss3)+loss_detection
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),loss_detection.unsqueeze(0)
        else:
            v = self.v_extractor(inputs, rois)
            rois,x = self.detection(v,None,rois,None,None)
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.5)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.tensor([]))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.5,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.tensor([])
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list
def multi_oicr(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Multi_OICR(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model
class Multi_OICR(nn.Module):
    def __init__(self, num_classes, block, layers):
        super().__init__()
        self.inplanes = 64
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        num_classes = 3
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")
        self.fpn = PyramidFeatures(128, 256, 512)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

            
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            loss = midn_loss + (loss1 + loss2 + loss3)
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0)
        else:
            rois = rois.squeeze().unsqueeze(0)
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

            
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x1 = self.icr1(v, v, v, rois, num) 
            x2 = self.icr2(v, v, v, rois, num) 
            x3 = self.icr3(v, v, v, rois, num) 
            x = (x1+x2+x3)/3
            x, rois = x[0], rois[0].cuda()
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.5)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes

    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.empty(0,5))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.5,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.empty(0,5).cuda()
            for i in range(n):
                X0 = float(int(data[i][0]))
                Y0 = float(int(data[i][1]))
                X1 = float(int(data[i][0] + data[i][2]))
                Y1 = float(int(data[i][1] + data[i][3]))
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list

class SLV_Retina(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmap_extractor = fmap_extractor()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        num_classes = 3
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.fpn = PyramidFeatures(128, 256, 512)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()

    def forward(self, inputs, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            f2,f3,f4 = self.fmap_extractor(inputs)
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels) 

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(inputs)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list)
            c_loss = c_loss.mean()
            r_loss = r_loss.mean()
            #print(f'm={midn_loss:.6f},l1={loss1:.6f},l2={loss2:.6f},l3={loss3:.6f},cl={c_loss},rl={r_loss}')
            loss = midn_loss + (loss1 + loss2 + loss3)+c_loss+r_loss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(c_loss+r_loss).unsqueeze(0)
        else:
            f2,f3,f4 = self.fmap_extractor(inputs)
            features = self.fpn([f2, f3, f4])

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(inputs)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
        

    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.empty(0,5))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.5,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.empty(0,5).cuda()
            for i in range(n):
                X0 = float(int(data[i][0]))
                Y0 = float(int(data[i][1]))
                X1 = float(int(data[i][0] + data[i][2]))
                Y1 = float(int(data[i][1] + data[i][3]))
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list

class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        #self.freeze_bn()
        
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(512*(14), 3))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, image, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            n = labels[:,3].sum()
            bs = labels.shape[0]
            rois = rois.squeeze()
            img_batch = image
        

            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            aux = self.aux(f4)
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f)
            x, midn_loss = self.midn(v, labels, num,aux)
            #d = (x.sum(dim=1)).clone().detach()
            #w = ((d*labels[:,:-1]).sum())/max(bs-n,1)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = generate_gt_retina(x3,rois,labels)#self.generate_gt_slv(x1,x2,x3,rois,labels) #

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(img_batch)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list)
            c_loss = c_loss.mean()#*w
            r_loss = r_loss.mean()/10#*w*w*w
            #print(f'm={midn_loss:.6f},l1={loss1:.6f},l2={loss2:.6f},l3={loss3:.6f},cl={c_loss},rl={r_loss}')
            loss = midn_loss + (loss1 + loss2 + loss3)+c_loss+r_loss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(c_loss+r_loss).unsqueeze(0)
           
        else:
            img_batch = image
            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            aux = self.aux(f4)
            print(F.sigmoid(aux))

            features = self.fpn([f2, f3, f4])
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            anchors = self.anchors(img_batch)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.2)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
        
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.empty(0,5))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.7,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.empty(0,5).cuda()
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list

class ResNet_easier(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        #self.freeze_bn()
        
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        self.midn = MIDN()
        
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(512*(14), 3))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, image, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            n = labels[:,3].sum()
            bs = labels.shape[0]
            rois = rois.squeeze()
            img_batch = image
        

            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            aux = self.aux(f4)
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f)
            x, midn_loss = self.midn(v, labels, num,aux)
            gt_list = self.generate_gt_slv(x,x,x,rois,labels) #generate_gt_retina(x3,rois,labels)#

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(img_batch)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list)
            c_loss = c_loss.mean()#*w
            r_loss = r_loss.mean()/10#*w*w*w
            loss = midn_loss +c_loss+r_loss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),(c_loss+r_loss).unsqueeze(0)
           
        else:
            img_batch = image
            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

            features = self.fpn([f2, f3, f4])
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            anchors = self.anchors(img_batch)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.2)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
        
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.empty(0,5))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.7,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.empty(0,5).cuda()
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def resnet18_easier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_easier(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def resnet18_fix(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fix(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def resnet18_full(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_full(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

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
        l = torch.ones((rois.shape[0],1))*label
        rois = torch.cat([rois,l.cuda()],dim=1)
        scores = scores[:i]
        gt_list.append(rois.clone().detach())
    return gt_list

class ResNet_fix(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        #self.freeze_bn()
        
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        self.midn = MIDN()
        self.icr1 = ICR()
        self.icr2 = ICR()
        self.icr3 = ICR()
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, image, labels, rois, num):
        if self.training:
            labels = labels.squeeze()
            rois = rois.squeeze()
            img_batch = image
            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)

            loss = midn_loss + (loss1 + loss2 + loss3)
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0)
           
        else:
            labels = labels.squeeze()
            rois = rois.squeeze()
            img_batch = image
        

            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x = self.midn(v, labels, num)
            x1 = self.icr1(v, x, labels, rois, num)
            x2 = self.icr2(v, x1, labels, rois, num)
            x3 = self.icr3(v, x2, labels, rois, num)
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels)#self.generate_gt_slv(x1,x2,x3,rois,labels) 

            return gt_list
        
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone()
            label = torch.max(labels_list[batch],0)[1]
            if label == 3:
                gt_list.append(torch.empty(0,5))
                continue
            rois = rois_list[batch]
            score_list = []
            
            for score in [x1[batch],x2[batch],x3[batch]]:
                score_list.append(score[:,label])
            score = (score_list[0]+score_list[1]+score_list[2])/3
            index = nms(rois,score,0.5)
            rois = rois[index]
            score = score[index]
            for r in range(len(rois)):
                mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = torch.where(mat>0.7,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = torch.empty(0,5).cuda()
            for i in range(n):
                X0 = float(int(data[i][0]))
                Y0 = float(int(data[i][1]))
                X1 = float(int(data[i][0] + data[i][2]))
                Y1 = float(int(data[i][1] + data[i][3]))
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if boxes.shape[0] == 0:
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            gt_list.append(boxes.clone().detach())
        return gt_list


class ResNet_full(nn.Module):
    def __init__(self, num_classes, block, layers):
        super().__init__()
        self.pre = resnet18_fix(3,pretrained=True)
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = focalloss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img,annot):
        self.pre.eval()
        if self.training:
            try:
                img_batch, annotations = img, annot
            except:
                print('failed')
                img_batch, annotations = data["img"].cuda().float(), self.pre(data["img"].cuda().float(),labels,rois,num)


        else:
            img_batch = img #eval img==data
        

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)
        
        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

