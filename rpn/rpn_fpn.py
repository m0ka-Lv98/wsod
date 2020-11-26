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
    def __init__(self, din):
        super(_RPN_FPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.nc_score_out = 1 * len(self.anchor_ratios) # 2(bg/fg) * 3 (anchor ratios) * 1 (anchor scale)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
        prior = 0.01
        self.RPN_cls_score.weight.data.fill_(0)
        self.RPN_cls_score.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # define anchor box offset prediction layer
        # self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.nc_bbox_out = 1 * len(self.anchor_ratios) * 4 # 4(coords) * 3 (anchors) * 1 (anchor scale)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)
        self.RPN_bbox_pred.weight.data.fill_(0)
        self.RPN_bbox_pred.bias.data.fill_(0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer_FPN(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer_FPN(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.anchors = 'none'

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, rpn_feature_maps, im_info, gt_boxes='none', num_boxes=0):        


        n_feat_maps = len(rpn_feature_maps)

        rpn_cls_scores = []
        rpn_cls_probs = []
        rpn_bbox_preds = []
        rpn_shapes = []

        for i in range(n_feat_maps):
            feat_map = rpn_feature_maps[i]
            batch_size = feat_map.size(0)
            
            # return feature map after convrelu layer
            rpn_conv1 = F.relu(self.RPN_Conv(feat_map), inplace=True)
            # get rpn classification score
            rpn_cls_score = self.RPN_cls_score(rpn_conv1)

            # get rpn offsets to the anchor boxes
            rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

            rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3]])
            rpn_cls_scores.append(rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1))
            rpn_cls_probs.append(torch.sigmoid(rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)))
            rpn_bbox_preds.append(rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))

        rpn_cls_score_alls = torch.cat(rpn_cls_scores, 1)
        rpn_cls_prob_alls = torch.cat(rpn_cls_probs, 1)
        rpn_bbox_pred_alls = torch.cat(rpn_bbox_preds, 1)

        n_rpn_pred = rpn_cls_score_alls.size(1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                 im_info, cfg_key, rpn_shapes))
        
        if isinstance(self.anchors,str):
            self.anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self.anchor_ratios, 
                    rpn_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type(torch.float)

        return rois[:,:,1:], rpn_cls_prob_alls, rpn_bbox_pred_alls,self.anchors.unsqueeze(0).cuda()