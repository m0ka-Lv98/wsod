import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .config import cfg
from .proposal_layer_fpn import _ProposalLayer_FPN
from .anchor_target_layer_fpn import _AnchorTargetLayer_FPN
from .net_utils import _smooth_l1_loss
from .generate_anchors import generate_anchors, generate_anchors_all_pyramids

import numpy as np
import math
import pdb
import time

class _RPN_FPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, head=False):
        super(_RPN_FPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_ratios = 3
        self.anchor_scales = 3
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE
        if isinstance(head,bool):
            self.head = nn.Module()
            self.head.conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
            self.nc_score_out = 18#1 * len(self.anchor_ratios) # 2(bg/fg) * 3 (anchor ratios) * 1 (anchor scale)
            self.head.cls_logits = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
            self.nc_bbox_out = 36#1 * len(self.anchor_ratios) * 4 # 4(coords) * 3 (anchors) * 1 (anchor scale)
            self.head.bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)
            self.head.bbox_pred.weight.data.fill_(0)
            self.head.bbox_pred.bias.data.fill_(0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer_FPN(self.feat_stride, self.anchor_scales, self.anchor_ratios)

    def forward(self, rpn_feature_maps, im_info):        
        n_feat_maps = len(rpn_feature_maps)

        rpn_cls_scores = []
        rpn_cls_probs = []
        rpn_bbox_preds = []
        rpn_shapes = []

        for i in range(n_feat_maps):
            feat_map = rpn_feature_maps[i]
            batch_size = feat_map.size(0)
            
            # return feature map after convrelu layer
            rpn_conv1 = F.relu(self.head.conv(feat_map), inplace=True)
            # get rpn classification score
            rpn_cls_score = self.head.cls_logits(rpn_conv1)

            # get rpn offsets to the anchor boxes
            rpn_bbox_pred = self.head.bbox_pred(rpn_conv1)

            rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3]])
            rpn_cls_scores.append(rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
            rpn_cls_probs.append(F.softmax(rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2),-1))
            rpn_bbox_preds.append(rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))

        rpn_cls_score_alls = torch.cat(rpn_cls_scores, 1)
        rpn_cls_prob_alls = torch.cat(rpn_cls_probs, 1)
        rpn_bbox_pred_alls = torch.cat(rpn_bbox_preds, 1)

        n_rpn_pred = rpn_cls_score_alls.size(1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                 im_info, cfg_key, rpn_shapes))
        
        return rois[:,:,1:], rpn_cls_prob_alls, rpn_bbox_pred_alls