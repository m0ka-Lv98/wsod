import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from .config import cfg
from torchvision.ops import nms
import time
from utils.utils import Anchors,BBoxTransform,ClipBoxes

import pdb

DEBUG = False

class _ProposalLayer_FPN(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer_FPN, self).__init__()
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clip = ClipBoxes()

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, :, 0]  # batch_size x num_rois x 1
        bbox_deltas = input[1]      # batch_size x num_rois x 4
        im_info = input[2]
        cfg_key = input[3]
        feat_shapes = input[4]        

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        anchors = self.anchors(torch.zeros((1,1,512,512)))
        anchors = anchors.squeeze()
        num_anchors = anchors.size(0)

        anchors = anchors.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)
        
        # Convert anchors into proposals via bbox transformations
        proposals = self.regressBoxes(anchors, bbox_deltas)
        

        # 2. clip predicted boxes to image
        proposals = self.clip(proposals, torch.zeros((1,1,512,512)))
        
        keep_idx = torch.nonzero(self._filter_boxes(proposals, 50),as_tuple=False)
        #print(keep_idx)
        #print(keep_idx.shape,proposals.shape,scores.shape)
        scores_keep = scores
        proposals_keep = proposals


        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            proposals_single = proposals_keep[i]
            _scores_single = scores_keep[i]

            
            scores_single = scores_keep[i]
            
            #k = keep_idx[keep_idx[:,0]==i,1]
            #scores_single = torch.zeros(_scores_single.shape).cuda().float()
            #scores_single[k] = _scores_single[k]


            


            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_single.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            s = time.time()
            keep_idx_i = nms(proposals_single, scores_single.squeeze(), 0.7)
            e = time.time()
            #print(f'nms {e-s}')
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single


        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size) & (hs >= min_size))
        return keep
