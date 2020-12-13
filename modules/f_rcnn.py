from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.detection as detection
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
import torchvision.models.detection.backbone_utils as backbone_utils
#from torchvision.models.detection._utils import overwrite_eps
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone#, _validate_resnet_trainable_layers
import numpy as np
import time
import matplotlib.pyplot as plt

from modules.modules import *
from rpn.rpn_fpn import _RPN_FPN

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class FasterRCNN(nn.Module):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super().__init__()
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = backbone.out_channels
        rpn_head = RPNHead(out_channels, 3)
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],output_size=7,sampling_ratio=2)

        resolution = 7
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        self.backbone = backbone
        self._rpn = _RPN_FPN(256)
        #self.rpn.head = rpn_head
        self.roi_heads = nn.Module()
        self.roi_heads.box_head = box_head
        self.box_predictor = box_predictor
        #self.freeze_bn()
        self.midn = MIDN(1024)
        self.icr1 = ICR(1024)
        self.icr2 = ICR(1024)
        self.icr3 = ICR(1024)
        self.rpn_loss = F_RPNLoss()
        self.f_loss = F_Loss()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.pool = nn.AvgPool2d(2)
        self.anchors = Anchors()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


    def forward(self,image,labels):
        try:
            bs = labels.shape[0]
        except:
            bs = 1
        anchors = self.anchors(image)
        features = self.backbone(image)
        _features = [self.pool(features[key]) for key in features]
        rois,rpn_c,rpn_r = self._rpn(_features, np.array([512,512]))
        num=rois.shape[1]
        image_size = [(512,512)]
        _rois = [r for r in rois]
        f = self.box_roi_pool(features, _rois,image_size)
        if self.training:
            v = self.roi_heads.box_head(f)
            x, midn_loss = self.midn(v, labels, num,aux=None)
            x1, loss1 = self.icr1(v, x, labels, rois, num)
            x2, loss2 = self.icr2(v, x1, labels, rois, num)
            x3, loss3 = self.icr3(v, x2, labels, rois, num)
            gt_list = self.generate_gt_slv(x1,x2,x3,rois,labels)
            rpn_closs,rpn_rloss = self.rpn_loss(rpn_c, rpn_r, anchors, gt_list)
        
        c,r = self.box_predictor(v)
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
    
    def generate_gt_slv(self,x1,x2,x3,rois_list,labels_list):
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
            index = score>0.8
            score = score[index]
            rois = rois[index]
            s=time.time()
            mat = c_utils.compute_mat(mat,rois,score)
            e=time.time()
            #print(f'mat{e-s}')
            mat = mat/(mat.max()+1e-8)
            mat = np.where(mat>0.,1,0)
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



def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=4, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    # check default parameters and by default set it to 3 if possible
    trainable_backbone_layers = 5#_validate_resnet_trainable_layers(pretrained or pretrained_backbone, trainable_backbone_layers)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
        #overwrite_eps(model, 0.0)
    return model