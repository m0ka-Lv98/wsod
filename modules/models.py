import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

from .modules import *
from rpn.rpn_fpn import _RPN_FPN

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
        
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
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
            aux = self.aux(f4)

            
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num, aux)
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

        self.focalLoss = FocalLoss()

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
            midn_score = x.sum(dim=1) #bs,4
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels) 

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(inputs)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list,w=midn_score,SOFT=False)
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
            mat = self.mat.clone().cpu().detach().numpy()
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
            rois = rois[index].cpu().detach().numpy()
            score = score[index].cpu().detach().numpy()
            mat = compute_mat(mat,rois,score)
            mat = mat/(mat.max()+1e-8)
            mat = np.where(mat>0.7,mat,0)
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

class ResNet_midn(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            out = 512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            out = 2048
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

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
                                nn.Linear(out*(14), 3))

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
            gt_list = generate_gt_retina(x,rois,labels)#self.generate_gt_slv(x,x,x,rois,labels)#
            loss1,loss2,loss3=torch.tensor(0).cuda().float(),torch.tensor(0).cuda().float(),torch.tensor(0).cuda().float()
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(img_batch)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list,SOFT=False)
            c_loss = c_loss.mean()
            r_loss = r_loss.mean()/10
            #print(f'm={midn_loss:.6f},l1={loss1:.6f},l2={loss2:.6f},l3={loss3:.6f},cl={c_loss},rl={r_loss}')
            loss = midn_loss + (loss1 + loss2 + loss3)+c_loss+r_loss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(c_loss+r_loss).unsqueeze(0)
           
        else:
            #rois = rois.squeeze().unsqueeze(0)
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

            '''aux = self.aux(f4)
            print(F.sigmoid(aux))
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f)
            x = self.midn(v, labels, num,0)
            gt_list = self.generate_gt_slv(x,x,x,rois,labels)'''

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
            mat = torch.where(mat>0.1,mat,self.zero).cpu().detach().numpy()
            heatmap = np.uint8(255*mat)
            if not self.training:
                plt.imshow(heatmap)
                plt.show()
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = 'none'
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if isinstance(boxes,str):
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            if isinstance(boxes,str):
                gt_list.append(boxes)
            else:
                gt_list.append(boxes.clone().detach())
        return gt_list

class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            out = 512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            out = 2048
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

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
                                nn.Linear(out*(14), 4))

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
            midn_score = x.sum(dim=1) #bs,4
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels)#generate_gt_retina(x3,rois,labels)#

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

            anchors = self.anchors(img_batch)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list,w=midn_score)
            c_loss = c_loss.mean()
            r_loss = r_loss.mean()
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
            #aux = self.aux(f4)
            #print(F.sigmoid(aux))

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

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
        
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
        bs,proposal,_ = x1.shape
        gt_list = []
        for batch in range(bs):
            mat = self.mat.clone().cpu().detach().numpy()
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
            rois = rois[index].cpu().detach().numpy()
            score = score[index].cpu().detach().numpy()
            mat = c_utils.compute_mat(mat,rois,score)
            #for r in range(len(rois)):
            #    mat[int(rois[r,1]):int(rois[r,3]),int(rois[r,0]):int(rois[r,2])] += score[r]
            mat = mat/(mat.max()+1e-8)
            mat = np.where(mat>0.7,mat,0)
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = 'none'
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if isinstance(boxes,str):
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            if isinstance(boxes,str):
                gt_list.append(boxes)
            else:
                gt_list.append(boxes.clone().detach())
        return gt_list

class ResNet_rpn(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            out = 512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            out = 2048
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

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
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(out*(14), 4))

        self.RCNN_rpn = _RPN_FPN(256)
        self.rpn_loss = RPNLoss()
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

    def forward(self, image, labels, rois,dum):
        if self.training:
            labels = labels.squeeze(1).squeeze(1)
            n = labels[:,3].sum()
            bs = labels.shape[0]
            rois = rois.squeeze()
            img_batch = image
            anchors = self.anchors(img_batch)

            s = time.time()
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
            
            e = time.time()
            #print(f'feature pyramid {e-s}')
            rois,rpn_c,rpn_r = self.RCNN_rpn(features, np.array([512,512]))
            #return rois
            s = time.time()
            #print(f'rpn {s-e}')
            num = rois.shape[1]
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            e = time.time()
            #print(f'roi pool {e-s}')
            v = self.feature_vector(f)
            s = time.time()
            #print(f'feature vector {s-e}')
            x, midn_loss = self.midn(v, labels, num,aux)
            midn_score = x.sum(dim=1) #bs,4
            e = time.time()
            #print(f'midn {e-s}')
            x1, loss1 = self.icr1(v, x/(x.max(dim=1,keepdim=True)[0])*x.sum(dim=1,keepdim=True), labels, rois, num)
            s = time.time()
            #print(f'icr1 {s-e}')
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            e = time.time()
            #print(f'icr2 {e-s}')
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            s = time.time()
            #print(f'icr3 {s-e}')
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels)#generate_gt_retina(x3,rois,labels)#
            e = time.time()
            #print(f'generate gt {e-s}')
            rpn_closs,rpn_rloss = self.rpn_loss(rpn_c, rpn_r, anchors, gt_list,w=midn_score)
            s = time.time()
            
            #print(f'compute rpn loss {s-e}')
    
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            e = time.time()
            #print(f'compute retina {e-s}')
            anchors = self.anchors(img_batch)
            #print(gt_list)
            c_loss,r_loss = self.focalLoss(classification, regression, anchors, gt_list,w=midn_score)
            s = time.time()
            #print(f'compute retina loss {s-e}')
            c_loss = c_loss.mean()
            r_loss = r_loss.mean()
            rpn_closs = rpn_closs.mean()
            rpn_rloss = rpn_rloss.mean()
            #print(f'm={midn_loss:.6f},l1={loss1:.6f},l2={loss2:.6f},l3={loss3:.6f},cl={c_loss},rl={r_loss}')
            loss = midn_loss + (loss1 + loss2 + loss3)+c_loss+r_loss+rpn_closs+rpn_rloss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(c_loss+r_loss).unsqueeze(0),(rpn_closs+rpn_rloss).unsqueeze(0)
           
        else:
            img_batch = image
            anchors = self.anchors(img_batch)
            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            aux = self.aux(f4)
            #print(F.sigmoid(aux))

            features = self.fpn([f2, f3, f4])
            rois,rpn_c,rpn_r= self.RCNN_rpn(features, np.array([512,512]))
            num = rois.shape[1]
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f)
            x = self.midn(v, labels, num,aux)
            print(x.sum(dim=1))
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
            mat = self.mat.clone().cpu().detach().numpy()
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
            rois = rois[index].cpu().detach().numpy()
            score = score[index].cpu().detach().numpy()
            mat = c_utils.compute_mat(mat,rois,score)
            mat = mat/(mat.max()+1e-8)
            mat = np.where(mat>0.7,mat,0)
            heatmap = np.uint8(255*mat)
            LAB = cv2.connectedComponentsWithStats(heatmap)
            n = LAB[0] - 1
            data = np.delete(LAB[2], 0, 0)
            boxes = 'none'
            for i in range(n):
                X0 = data[i][0]
                Y0 = data[i][1]
                X1 = data[i][0] + data[i][2]
                Y1 = data[i][1] + data[i][3]
                if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                    continue
                if isinstance(boxes,str):
                    boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
                else:
                    boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
            if isinstance(boxes,str):
                gt_list.append(boxes)
            else:
                gt_list.append(boxes.clone().detach())
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

        self.focalLoss = FocalLoss()

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

        self.focalLoss = FocalLoss()

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

    def forward(self, img, annot=None, soft=False):
        
        if self.training:
            img_batch, annotations = img, annot
            
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
            return self.focalLoss(classification, regression, anchors, annotations, SOFT=soft)
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

def multi_oicr(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Multi_OICR(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def multi_midn(num_classes=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Multi_MIDN(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

class Multi_MIDN(nn.Module):
    def __init__(self, num_classes, block, layers):
        super().__init__()
        self.inplanes = 64
        self.midn = MIDN()
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
        
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            nn.Flatten(),
                                            nn.Linear(256*(5), 2048),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(2048),
                                            nn.Linear(2048, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
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
            aux = self.aux(f4)

            
            features = self.fpn([f2, f3, f4])
            f = self.multi_scale_roi_pool(features[0],features[1],features[2], rois)
            v = self.feature_vector(f) 
            x, midn_loss = self.midn(v, labels, num, aux)
            loss1,loss2,loss3 = torch.tensor(0).cuda().float(),torch.tensor(0).cuda().float(),torch.tensor(0).cuda().float()
            loss = midn_loss + (loss1 + loss2 + loss3)
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0)
        else:
            rois = rois.squeeze().unsqueeze(0)
            num = rois.shape[1]
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
            x = self.midn(v, 0, num, 0)
            x = x/x.max()
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

def resnet_midn(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_midn(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=dl_root), strict=False)
    return model

def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=dl_root), strict=False)
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

def resnet_rpn(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_rpn(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

class Faster_RCNN(nn.Module):
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
            out = 512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            out = 2048
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        self.feature_vector = nn.Sequential(ASPP([1,2,3]),
                                            nn.Flatten(),
                                            nn.Linear(256*(14), 2048),
                                            #nn.BatchNorm1d(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 1024),
                                            #nn.BatchNorm1d(1024),
                                            nn.ReLU(inplace=True))
        self.multi_scale_roi_pool = MultiScaleROIPool()
        self.midn = MIDN(1024)
        self.icr1 = ICR(1024)
        self.icr2 = ICR(1024)
        self.icr3 = ICR(1024)
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(out*(14), 4))

        self.RCNN_rpn = _RPN_FPN(256)
        self.rpn_loss = F_RPNLoss()
        self.f_loss = F_Loss()
        #self.rpn_loss = softRPNLoss()
        #self.f_loss = FocalLoss()
        self.RCNN_c = nn.Sequential(nn.Linear(256*4, 4),nn.Softmax(-1))
        self.RCNN_c[0].weight.data.fill_(0)
        self.RCNN_c[0].bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.RCNN_r = nn.Linear(256*4, 4)
        self.RCNN_r.weight.data.fill_(0)
        self.RCNN_r.bias.data.fill_(0)
        #self.freeze_bn()

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

    def forward(self, image, labels,rois=None,n=None):
        try:
            bs = labels.shape[0]
        except:
            bs = 1
        img_batch = image
        anchors = self.anchors(img_batch)

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

        rois,rpn_c,rpn_r = self.RCNN_rpn(features, np.array([512,512]))
        num = rois.shape[1]

        f = self.multi_scale_roi_pool(features, rois)
        v = self.feature_vector(f)
        if self.training:
            #return rois
            x, midn_loss = self.midn(v, labels, num,aux=None)
            midn_score = x.sum(dim=1)
            #print(x.shape,(x.sum(1)).shape,(x.max(1,keepdim=True)[0]))
            x = x/x.max(1,keepdim=True)[0]*x.sum(1,keepdim=True)
            loss1,loss2,loss3 = torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = generate_gt_slv(x1,x2,x3,rois,labels)
            rpn_closs,rpn_rloss = self.rpn_loss(rpn_c, rpn_r, anchors, gt_list)

        c = self.RCNN_c(v)
        r = self.RCNN_r(v)
        c = c.view(bs,num,-1)
        r = r.view(bs,num,-1)
        if self.training:
            closs,rloss = self.f_loss(c, r, rois, gt_list)
            #return gt_list
            closs = closs.mean()
            rloss = rloss.mean()
            rpn_closs = rpn_closs.mean()
            rpn_rloss = rpn_rloss.mean()
            print(f'f_c={closs:.4f},f_r={rloss:.4f},rpn_c={rpn_closs:.4f},rpn_r={rpn_rloss:.4f}')
            loss = midn_loss + (loss1 + loss2 + loss3)+closs+rloss+rpn_closs+rpn_rloss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(closs+rloss).unsqueeze(0),(rpn_closs+rpn_rloss).unsqueeze(0)
           
        else:
            x = c
            rois = self.regressBoxes(rois, r)
            rois = self.clipBoxes(rois, img_batch) 
            rois = rois[0]
            x = x[0]
            print(x)
            print(x.max(dim=0))
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                print('no')
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.2)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes


def faster_rcnn18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Faster_RCNN(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    dl_root = "/data/unagi0/masaoka/resnet_model_zoo/"
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def faster_rcnn34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Faster_RCNN(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    dl_root = "/data/unagi0/masaoka/resnet_model_zoo/"
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=dl_root), strict=False)
    return model

def faster_rcnn50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Faster_RCNN(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=dl_root), strict=False)
    return model

class Fast_RCNN(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            out = 512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            out = 2048
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        #self.freeze_bn()

        
        
        self.feature_vector = nn.Sequential(ASPP([1,2,3]),
                                            nn.Flatten(),
                                            nn.Linear(256*(14), 2048),
                                            nn.BatchNorm1d(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 1024),
                                            nn.BatchNorm1d(1024),
                                            nn.ReLU(inplace=True))
        
        self.multi_scale_roi_pool = MultiScaleROIPool()
        self.midn = MIDN(1024)
        self.icr1 = ICR(1024)
        self.icr2 = ICR(1024)
        self.icr3 = ICR(1024)
        self.aux = nn.Sequential(ASPP([1,2,3]),
                                nn.Flatten(),
                                nn.Linear(out*(14), 4))

        self.f_loss = F_Loss()
        self.mat = nn.Parameter(torch.zeros((512,512))).requires_grad_(False)
        self.zero = nn.Parameter(torch.tensor([0.])).requires_grad_(False)
        self.RCNN_c = nn.Sequential(nn.Linear(256*4, 4),nn.Softmax(-1))
        self.RCNN_c[0].weight.data.fill_(0)
        self.RCNN_c[0].bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.RCNN_r = nn.Linear(256*4, 4)
        self.RCNN_r.weight.data.fill_(0)
        self.RCNN_r.bias.data.fill_(0)

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

    def forward(self, image, labels, rois):
        try:
            bs = labels.shape[0]
        except:
            bs = 1
        img_batch = image
        anchors = self.anchors(img_batch)

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
        
        num = rois.shape[1]
        
        f = self.multi_scale_roi_pool(features, rois)
        v = self.feature_vector(f)
        if self.training:
            x, midn_loss = self.midn(v, labels, num,aux=None)
            midn_score = x.sum(dim=1) #bs,4
            x1, loss1 = self.icr1(v, x, labels, rois, num)#/(x.max(dim=1,keepdim=True)[0])*x.sum(dim=1,keepdim=True)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = generate_gt_slv(x1,x2,x3,rois,labels)#generate_gt_retina(x3,rois,labels)#
            #return gt_list
        c = self.RCNN_c(v)
        r = self.RCNN_r(v)
        c = c.view(bs,num,-1)
        r = r.view(bs,num,-1)
        if self.training:
            closs,rloss = self.f_loss(c, r, rois, gt_list)
            closs = closs.mean()
            rloss = rloss.mean()
            print(f'f_c={closs:.4f},f_r={rloss:.4f}')
            loss = midn_loss + (loss1 + loss2 + loss3)+closs+rloss
            return x, loss.unsqueeze(0),midn_loss.unsqueeze(0),loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),(closs+rloss).unsqueeze(0)
           
        else:
            x = c
            print(rois.shape,r.shape)
            rois = self.regressBoxes(rois, r)
            rois = self.clipBoxes(rois, img_batch) 
            rois = rois[0]
            x = x[0]
            rois = rois[torch.max(x,1)[1]!=3]
            x = x[torch.max(x,1)[1]!=3]
            if x.size() == torch.Size([0,4]):
                return torch.tensor([]),torch.tensor([]),torch.tensor([])
            classes = torch.max(x,1)[1] #proposals
            scores = torch.max(x,1)[0] #proposals
            index = nms(rois,scores,0.2)
            scores = scores[index]
            labels = classes[index]
            bboxes = rois[index]
            return scores, labels, bboxes

def fast_rcnn18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Fast_RCNN(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    dl_root = "/data/unagi0/masaoka/resnet_model_zoo/"
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=dl_root), strict=False)
    return model

def generate_gt_slv(x1,x2,x3,rois_list,labels_list):
    bs,proposal,_ = x1.shape
    gt_list = []
    for batch in range(bs):
        mat = torch.zeros((512,512)).float().numpy()
        label = torch.max(labels_list[batch],0)[1]
        if label == 3:
            gt_list.append(torch.empty(0,5))
            continue
        rois = rois_list[batch]
        score_list = []
            
        for score in [x1[batch],x2[batch],x3[batch]]:
            score_list.append(score[:,label])
        score = (score_list[0]+score_list[1]+score_list[2])/3
        index = nms(rois,score,0.7)
        rois = rois[index].cpu().detach().numpy()
        score = score[index].cpu().detach().numpy()
        index = score>0.1
        score = score[index]
        rois = rois[index]
        s=time.time()
        mat = compute_mat(mat,rois,score)
        e=time.time()
        #print(f'mat{e-s}')
        mat = mat/(mat.max()+1e-8)
        mat = np.where(mat>0.5,1,0)
        heatmap = np.uint8(255*mat)
        LAB = cv2.connectedComponentsWithStats(heatmap)
        n = LAB[0] - 1
        data = np.delete(LAB[2], 0, 0)
        boxes = 'none'
        for i in range(n):
            X0 = data[i][0]
            Y0 = data[i][1]
            X1 = data[i][0] + data[i][2]
            Y1 = data[i][1] + data[i][3]
            if abs(X0-X1) < 50 or abs(Y0-Y1) < 50:
                continue
            if isinstance(boxes,str):
                boxes = torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()
            else:
                boxes = torch.cat((boxes, torch.tensor([[X0,Y0,X1,Y1,label]]).cuda()), dim=0)
        if isinstance(boxes,str):
            gt_list.append(boxes)
        else:
            gt_list.append(boxes.clone().detach())
    return gt_list
