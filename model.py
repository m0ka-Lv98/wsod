import torch.nn as nn
import torch
import math
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numbers
from gradcam import *
from losses import *
import losses

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=3, bias=True))
        self.gradcam = GradCam(model=self.resnet50, feature_module=self.resnet50.layer4, target_layer_names=["2"], use_cuda=True)
    def forward(self,x,e=False, aug=False):
        if not e:
            x = self.resnet50(x)
            return x

        else:
            #x 1,3,512,512
            if aug:
                masks = augmented_grad_cam(self.gradcam, x) 
            else:
                masks = self.gradcam(x,None)
            boxes = torch.tensor([])
            scores = torch.tensor([])
            classification = torch.tensor([])
            for threshold in reversed(range(11)):
                threshold = threshold/10
                for num, mask in enumerate(masks):
                    if isinstance(mask,numbers.Number):
                        continue
                    #mask 1,512,512
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

