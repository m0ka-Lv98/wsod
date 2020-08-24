
# In[1]:


import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from make_dloader import make_data
from torch.utils.data import DataLoader
from utils import bbox_collate, MixedRandomSampler
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
from model import ResNet50
import time
from scipy.ndimage import gaussian_filter
import random

config = yaml.safe_load(open('./config.yaml'))
dataset_means = json.load(open(config['dataset']['mean_file']))


# In[2]:


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


# In[3]:


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


# In[4]:


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
        #print(label)
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
    
        return label, cam_list



model = ResNet50()
model.cuda()
model.load_state_dict(torch.load("/data/unagi0/masaoka/wsod/model/resnet50_classify1.pt"))


grad_cam = GradCam(model=model.resnet50, feature_module=model.resnet50.layer4, target_layer_names=["2"], use_cuda=True)

_, dataset_val, _ = make_data()

size = config["inputsize"]
val = config["dataset"]["val"]
val.sort()
val = ''.join(map(str,val))


# In[16]:


val_anomaly = dataset_val.with_annotation()
#dataset_val = val_anomaly
dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=bbox_collate)
unnormalize = transf.UnNormalize(dataset_means['mean'], dataset_means['std'])
normalize = transf.Normalize(dataset_means['mean'], dataset_means['std'])


class loss_r(nn.Module):
    def __init__(self):
        super(loss_r, self).__init__()
        
    def forward(self, i, target, rho = 1e5):
        i = i.cpu().data.numpy()
        target = target.cpu().data.numpy()
        i = np.where(i>0.4,i,0)
        target = np.where(target>0.4,target,0)
        loss = 255*abs(target-i)/i.sum()
        loss = loss.sum()
        return 0.5*loss*rho
    
class loss_l2(nn.Module):
    def __init__(self):
        super(loss_l2, self).__init__()
    
    def forward(self, i, rho = 1e-4):
        loss = (255*i)**2
        loss = loss.sum()
        return 0.5*rho*loss
    
class loss_tv(nn.Module):
    
    def __init__(self):
        super(loss_tv, self).__init__()
    
    def forward(self, i, rho = 3000):
        w, h = i.shape[0], i.shape[1]
        lx = i[1:, :h-1] - i[ :w-1, :h-1]
        ly = i[:w-1, 1:] - i[:w-1, :h-1]
        lx, ly = abs(lx), abs(ly)
        loss = lx.sum()+ly.sum()
        return rho*loss
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.l2loss = loss_l2()
        self.TVloss = loss_tv()
        self.r_loss = loss_r()
        
    def forward(self, i, t):
        l2loss = self.l2loss(i)
        TVloss = self.TVloss(i)
        r_loss = self.r_loss(i,t)
        loss = l2loss + TVloss + r_loss
        return TVloss, r_loss, l2loss
def converge_map(masks):
    #init 512,512 ndarray ; mask b,512,512 ndarray
    seq1 = iaa.Sequential([
                    iaa.Affine(
        rotate=iap.DiscreteUniform(-180,179)*(-1)
                    )])
    ia.seed(0)
    masks = seq1(images=masks)
    
    mask_reconvert = torch.from_numpy(masks) #b,512,512 tensor
    mp = mask_reconvert[0]
    calc_loss = Loss()
    calc_loss.cuda()
    losses = []
    TVloss, r_loss, l2 = calc_loss(mp.cuda(), mask_reconvert.cuda())
    l2 = l2/(size**2)
    TVloss = TVloss/(size**2)
    r_loss = r_loss/masks.shape[0]/(size**2)
    print(f"TV:{TVloss}, r:{r_loss}, l2:{l2}")
    conf_loss = r_loss+TVloss
    mp = mp.data.numpy()
    return mp, conf_loss #512,512 ndarray
def augmented_grad_cam(gcam, image):           #gcamはheatmapとlabelを出力するクラス
    #img.shape=B,C,H,W　tensor
    img = image.clone()
    img = img.squeeze().numpy().transpose(1,2,0)  #512,512,3
    b = 10
    img = np.tile(img,(b,1,1,1)) #b,512,512,3
    seq = iaa.Sequential([
                    iaa.Affine(
        rotate=iap.DiscreteUniform(-180, 179)
                    )])
    ia.seed(0)
    img = seq(images=img)
    img = torch.from_numpy(img)
    labels, masks = grad_cam(img.permute(0,3,1,2), None) #label [1,0,0,0] masks [(40,512,512), 0, (40,512,512), 0]
    maps = []
    for i, label in enumerate(labels):
        if label == 0:
            maps.append(0)
            continue
        elif i == len(labels)-1:
            continue
        else:
            mp, _ = converge_map(masks[i])
            maps.append(mp[np.newaxis,:,:])
    
    return maps
    
class Conf(nn.Module):
    def __init__(self):
        super(Conf, self).__init__()
    def forward(self, x):
        converged, conf = converge_map(x.numpy())
        return converged, conf
    
def calc_conf(masks):
    reconvert = torch.from_numpy(masks) 
    #信頼度の計算をするクラスConf
    conf = Conf() 
    conf.cuda()
    mask, conf = conf(reconvert)
    return mask, conf

def high_conf_maps(gcam, image):           #gcamはheatmapとlabelを出力するクラス
    #img.shape=B,C,H,W　tensor
    img = image.clone()
    img = img.squeeze().numpy().transpose(1,2,0)  #512,512,3
    b = 20
    img = np.tile(img,(b,1,1,1)) #b,512,512,3
    seq = iaa.Sequential([
                    iaa.Affine(
        rotate=iap.DiscreteUniform(-180, 179)
                    )])
    s = iaa.Sequential([iaa.Affine(scale=0.9)])
    img = s(images = img)
    ia.seed(0)
    img = seq(images=img)
    img = torch.from_numpy(img)
    labels, masks = grad_cam(img.permute(0,3,1,2), None) #label [1,0,1] masks [(40,512,512), 0, (40,512,512)]

    maps = []
    eps = 35
    for i, label in enumerate(labels):
        if label == 0:
            maps.append(0)
            continue
        else:
            pseudo_map, conf = calc_conf(masks[i])
            print(f"conf:{conf}")
            if conf < eps:
                maps.append(pseudo_map)
            else:
                maps.append(0)
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
    for i in range(n):
    # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        if boxes.shape[0] == 0:
            boxes = torch.tensor([[x0,y0,x1,y1]])
        else:
            torch.cat((boxes, torch.tensor([[x0,y0,x1,y1]])), dim=0)
        score = threshold
        if not isinstance(img, numbers.Number):
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 0), thickness = 4)
    #if not isinstance(img, numbers.Number):
    #    plt.imshow(image)
    #    plt.show()
    return boxes, score

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


#通常の評価　マスクなし
threshold= .5
gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
ids = []
for idx, data in enumerate(dataloader_val): 
    image = data["img"].clone()
    print(f'evaluating {idx}/{len(dataloader_val)}', end = '\n')
    
    maps = high_conf_maps(grad_cam, image)
    image = data["img"].clone()
    for m in maps:
        if not isinstance(m, numbers.Number):
            ids.append(idx)
            break
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]
    
    
    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn=(1-output)*(1-target)
    tpa += tp
    fpa += fp
    tna += tn
    
    
path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='w') as f:
    f.write('w/o masks, searching confidential pseudo labels\n')
    f.write(f'id for pseudo: {ids}\n')
    f.write(f'TP: {tpa}, FN: {gt-tpa}, FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')

#通常の評価　d_val_conf のみ

gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
for idx, data in enumerate(dataloader_val): 
    if idx > ids[-1]:
        break
    if not (idx in ids):
        continue
    print(f'evaluating {idx}/{len(dataloader_val)}', end = '\r')
    for i in range(len(data["bboxes"][0])):
        x1 = int(data["bboxes"][0][i][0])
        y1 = int(data["bboxes"][0][i][1])
        x2 = int(data["bboxes"][0][i][2])
        y2 = int(data["bboxes"][0][i][3])

    image = data["img"].clone()
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]
    image = data["img"].clone()
    
    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn = (1 - output) * (1 - target)
    tpa += tp
    fpa += fp
    tna += tn
    
path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='a') as f:
    f.write('only pseudo labels, evaluating\n')
    f.write(f'TP: {tpa},FN: {gt-tpa},FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')


#腫瘍にマスク、正常画像はランダムにマスク
gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
all_masks = []
size = config["inputsize"]
for idx, data in enumerate(dataloader_val): 
    image = data["img"].clone()
    print(f'masking {idx}/{len(dataloader_val)}', end = '\r')
    mask = torch.ones((size,size))
    flag = -1
    for i in range(len(data["bboxes"][0])):
        flag = 1
        x1 = int(data["bboxes"][0][i][0])
        y1 = int(data["bboxes"][0][i][1])
        x2 = int(data["bboxes"][0][i][2])
        y2 = int(data["bboxes"][0][i][3])
        mask[y1:y2,x1:x2] = 0
        inv_mask = 1 - mask
        if (y2-y1)*(x2-x1) < size*size/4:
            all_masks.append(mask)
    if flag != 1:
        mask = random.choice(all_masks)
        inv_mask = 1 - mask
        
    blur = gaussian_filter(image*inv_mask,10)
    image_mask = image*mask
    image = image_mask+ blur
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]
    
    
    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn = (1 - output) * (1 - target)
    tpa += tp
    fpa += fp
    tna += tn
    
path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='a') as f:
    f.write('all images, masking anomaly\n')
    f.write(f'TP: {tpa},FN: {gt-tpa},FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')

#腫瘍にマスク、正常画像はランダムにマスク
gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
size = config["inputsize"]
for idx, data in enumerate(dataloader_val): 
    if idx > ids[-1]:
        break
    if not (idx in ids):
        continue
    image = data["img"].clone()
    print(f'masking {idx}/{len(dataloader_val)}', end = '\r')
    mask = torch.ones((size,size))
    flag = -1
    for i in range(len(data["bboxes"][0])):
        flag = 1
        x1 = int(data["bboxes"][0][i][0])
        y1 = int(data["bboxes"][0][i][1])
        x2 = int(data["bboxes"][0][i][2])
        y2 = int(data["bboxes"][0][i][3])
        mask[y1:y2,x1:x2] = 0
        inv_mask = 1 - mask
        
    if flag != 1:
        mask = random.choice(all_masks)
        inv_mask = 1 - mask
        
    blur = gaussian_filter(image*inv_mask,10)
    image_mask = image*mask
    image = image_mask+ blur
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]
    
    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn = (1 - output) * (1 - target)
    tpa += tp
    fpa += fp
    tna += tn
    
path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='a') as f:
    f.write('only confidential images, masking anomaly\n')
    f.write(f'TP: {tpa},FN: {gt-tpa},FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')

#腫瘍を切り出し表示、正常画像はランダムに切り出し表示
gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
all_masks = []
for idx, data in enumerate(dataloader_val): 
    image = data["img"].clone()
    print(f'emphasizing {idx}/{len(dataloader_val)}', end = '\r')
    mask = torch.zeros((size, size))
    flag = 0
    for i in range(len(data["bboxes"][0])):
        flag = 1
        x1 = int(data["bboxes"][0][i][0])
        y1 = int(data["bboxes"][0][i][1])
        x2 = int(data["bboxes"][0][i][2])
        y2 = int(data["bboxes"][0][i][3])
        mask[y1:y2,x1:x2] = 1
        inv_mask = 1 - mask
        if (y2-y1)*(x2-x1) < size*size/2:
            all_masks.append(mask) 

    if flag ==0:
        mask = random.choice(all_masks)
        inv_mask = 1 - mask
    blur = gaussian_filter(image*inv_mask,10)
    image = image*mask
    image = image + blur
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]
    
    
    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn = (1 - output) * (1 - target)
    tpa += tp
    fpa += fp
    tna += tn

        
path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='a') as f:
    f.write('all images, emphasizing anomaly\n')
    f.write(f'TP: {tpa},FN: {gt-tpa},FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')

#腫瘍を切り出し表示、正常画像はランダムに切り出し表示
gt, tpa, fpa, tna = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
for idx, data in enumerate(dataloader_val): 
    if not (idx in ids):
        continue
    image = data["img"].clone()
    print(f'emphasizing {idx}/{len(dataloader_val)}', end = '\r')
    mask = torch.zeros((size, size))
    flag = 0
    for i in range(len(data["bboxes"][0])):
        flag = 1
        x1 = int(data["bboxes"][0][i][0])
        y1 = int(data["bboxes"][0][i][1])
        x2 = int(data["bboxes"][0][i][2])
        y2 = int(data["bboxes"][0][i][3])
        mask[y1:y2,x1:x2] = 1
        inv_mask = 1 - mask

    if flag ==0:
        mask = random.choice(all_masks)
        inv_mask = 1 - mask
    blur = gaussian_filter(image*inv_mask,10)
    image = image*mask
    image = image + blur
    masks = grad_cam(image, None)
    target = np.zeros(3)
    if len(data["bboxes"][0]) != 0:
        target[int(data["labels"][0][0])] = 1
    output = masks[0]

    gt += target
    tp = (output * target)
    fp = (output * (1 - target))
    tn = (1 - output) * (1 - target)
    tpa += tp
    fpa += fp
    tna += tn

path = f'/data/unagi0/masaoka/wsod/text/val{val}.txt'
with open(path, mode='a') as f:
    f.write('only confidential images, emphasizing anomaly\n')
    f.write(f'TP: {tpa},FN: {gt-tpa},FP: {fpa},TN: {tna}\n')
    f.write(f'Precision: {tpa/(tpa+fpa+1e-10)}, Recall: {tpa/(gt+1e-10)}, Spec: {tna/(tna+fpa+1e-10)}\n')





