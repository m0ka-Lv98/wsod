import torch
import torch.nn as nn
from torchvision.ops import roi_align, nms
from utils import calc_iou
import torchvision.models as models
import yaml
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F

config = yaml.safe_load(open('./config.yaml'))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7,reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss
        if self.reduction=='none':
            return loss

        return loss.sum()


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
        layers = list(model.children())[:-2]
        self.feature_map = nn.Sequential(*layers)
        self.roi_pool = _ROIPool()
        #self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_vector = nn.Sequential(ASPP([1,2]),
                                            #nn.AdaptiveAvgPool2d(1),
                                            nn.Flatten(),
                                            nn.Linear(512*(5), 500),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm1d(500),
                                            nn.Linear(500, 500),
                                            nn.BatchNorm1d(500),
                                            nn.ReLU(inplace=True))
    def forward(self, inputs, rois):
        f = self.feature_map(inputs)
        f = self.roi_pool(f, rois)
        #f = self.gap(f).view(f.shape[0], f.shape[1]) #batch*proposal, ch
        f = self.feature_vector(f) 
        return f
    
class MIDN(nn.Module):
    """input: feature vector, labels
         output: scores per proposal, loss"""
    def __init__(self):
        super().__init__()
        c_in = 500
        self.layer_c = nn.Linear(c_in, 4)
        self.layer_d = nn.Linear(c_in, 4)
        self.softmax_c = nn.Softmax(dim=2)
        self.softmax_d = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inputs, labels, num):
        bs, proposal = inputs.shape[0]//num, num
        x_c = self.layer_c(inputs).view(bs, proposal, -1) #bs, proposal, 4
        x_d = self.layer_d(inputs).view(bs, proposal, -1)
        sigma_c = self.softmax_c(x_c)
        sigma_d = self.softmax_d(x_d)
        x_r = sigma_c * sigma_d #bs, proposal, 4
        phi_c = x_r.sum(dim=1) #bs, 4
        print((torch.max(x_r,1))[0].shape,phi_c.shape,x_r.shape)
        scaled = x_r/torch.max(x_r,1)[0]*phi_c
        if not self.training:
            print(scaled)
            return scaled
        loss = self.loss(phi_c, torch.max(labels,1)[1])
        return   scaled, loss#phi_c.unsqueeze(1), loss # sigma_c,loss#x_r,loss
        

class ICR(nn.Module):
    """input: feature vector (bs*proposal, ch)
                     k-1th proposal scores(bs, proposal,3or4)
                     supervision (label) (bs)
                     ROI proposals
         output: refined proposal scores, loss"""
    def __init__(self):
        super().__init__()
        c_in = 500
        self.I_t = 0.5
        self.fc = nn.Linear(c_in, 4)
        self.softmax = nn.Softmax(dim=2)
        #self.loss = nn.CrossEntropyLoss(reduction="none")
        self.loss = FocalLoss(gamma=2)
        """self.y_k = torch.zeros(bs, proposal, 4).cuda()
        self.y_k[:, :, 3] = 1
        self.w = torch.zeros(bs, proposal).cuda()"""
        
    def forward(self, inputs, pre_score, labels, rois, num):
        bs, proposal = inputs.shape[0]//num, num
        xr_k = self.fc(inputs).view(bs, proposal, -1) #bs, proposal, 4
        xr_k = self.softmax(xr_k)
        
        _xr_k = xr_k.view(bs*proposal, -1)
        self.y_k = torch.zeros(bs, proposal, 4).cuda()
        self.y_k[:, :, 3] = 1
        self.w = torch.zeros(bs, proposal).cuda()
        I = torch.zeros(bs, proposal)
        for batch in range(bs):
            for c in range(3):
                if labels[batch][c]:
                    #print(f'label{c}')
                    #m = torch.max(pre_score[batch, :, c], 0)
                    #x = m[0].item()
                    #j = m[1].item()
                    j_list = (pre_score[batch,:,c]>0.5).nonzero().squeeze()
                    x_list = pre_score[batch,j_list,c]
                    if (j_list).size() == torch.Size([0]) or (j_list).size() == torch.Size([1]) or (j_list).size() == torch.Size([]):
                        m = torch.max(pre_score[batch, :, c], 0)
                        x = m[0].item()
                        j = m[1].item()
                        x_list = [x]
                        j_list =[j]
                    mat = box_iou(rois[batch], rois[batch][j_list])
                    for i,j in enumerate(j_list):
                        for r in range(proposal):
                            _I = mat[r][i]
                            if _I > I[batch, r]:
                                I[batch, r] = _I
                                self.w[batch, r] = x_list[i]
                                if _I > self.I_t:
                                    #print(f'next supervision index{r}')
                                    self.y_k[batch, r, c] = 1
                                    self.y_k[batch, r, 3] = 0
        self.y_k = self.y_k.view(bs*proposal, -1)
        self.w = self.w.view(bs*proposal, 1)
        loss = self.loss(_xr_k.cuda().float(), torch.max(self.y_k, 1)[1])
        loss = torch.mean(self.w*loss)
        if not self.training:
            return xr_k
        return xr_k, loss

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
            v = self.v_extractor(inputs, rois)
            x, midn_loss = self.midn(v, labels, num)
            x, loss1 = self.icr1(v, x, labels, rois, num)
            x, loss2 = self.icr2(v, x, labels, rois, num)
            x, loss3 = self.icr3(v, x, labels, rois, num) 
            print(midn_loss,loss1,loss2,loss3)
            loss = midn_loss + loss1 + loss2 + loss3
            return x, loss.unsqueeze(0)  
        else:
            self.three = torch.tensor([3.]).cuda()
            self.inf = torch.tensor([-1.]).cuda()
            #rois = rois.squeeze()
            v = self.v_extractor(inputs, rois)
            x = self.midn(v, labels, num)
            x = self.icr1(v, x, labels, rois, num) 
            x, rois = x[0], rois[0].cuda()
            h = 0
            while(1):
                if torch.max(x[h],0)[1]==3:
                    x = torch.cat([x[:h],x[h+1:]])
                    rois = torch.cat([rois[:h],rois[h+1:]])
                else:
                    h+1
                if len(x)==h:
                    break
            print(x.size())
            if x.size() == torch.Size([0,4]):
                return [],[],[]
            s, i = torch.max(x, 1)
            s = torch.where(i==self.three,self.inf,s)
            sort = torch.argsort(s, descending=True)
            s, i = s.view(-1,1), i.view(-1,1).cuda().float()
            #print(s.shape, i.shape, rois.shape)
            cat = torch.cat([s, i ,rois], dim=1)
            cat = cat[sort, :]
            scores = cat[:, 0]
            labels = cat[:, 1]
            bboxes = cat[:, 2:]
            return scores, labels, bboxes
