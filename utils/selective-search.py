from dataset.dataset import *
from selective_search import selective_search
import numpy as np
import json
from collections import OrderedDict
import concurrent.futures

image_root = "/data/unagi0/masaoka/dataset/capsule/capsule_crop"
annotation_file = "/data/unagi0/masaoka/endoscopy/annotations/cleanup0906/capsule_cocoformat.json"
MODE = "quality"

d = MedicalBboxDataset(annotation = annotation_file, data_path = image_root)

def annotations(ids, bboxes):
    tmp = OrderedDict()
    tmp[f"p_bbox{ids}"] = bboxes
    return tmp

def impl_ss(num):
    tmp = OrderedDict()
    for i, imgids in enumerate(d.imgids):
        if imgids<num:
            continue
        if imgids>=num+5000:
            break
        print(i)
        x = d.load_image(i)
        print(f'sum={x.sum()}')
        boxes = selective_search(x, mode = MODE, random_sort = True)
        k = []
        for j,box in enumerate(boxes):
            ### 一定の長さが無い矩形表示しない
            if abs(box[2]-box[0]) > 50 and abs(box[3]-box[1])>50:
                print(box[2]-box[0],box[3]-box[1])
                k.append(j)
        #boxes = (np.array(boxes)[k]).tolist()
        tmp[f"p_bbox{imgids}"] = boxes
    return tmp
num=5000
js = OrderedDict()
result = impl_ss(num)
js["pseudo_annotations"] = result
with open(f"/data/unagi0/masaoka/endoscopy/annotations/pseudo_annotations{int(num//1000)}.json", "w") as f:
    json.dump(js, f, indent=2)