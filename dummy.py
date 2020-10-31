from dataset import *
from selective_search import selective_search
import numpy as np
import json
from collections import OrderedDict
import concurrent.futures

image_root = "/data/unagi0/masaoka/dataset/capsule/capsule_crop"
annotation_file = "/data/unagi0/masaoka/endoscopy/annotations/cleanup0906/capsule_cocoformat.json"
MODE = "fast"

d = MedicalBboxDataset(annotation = annotation_file, data_path = image_root)
def annotations(ids, bboxes):
    tmp = OrderedDict()
    tmp[f"p_bbox{ids}"] = bboxes
    return tmp

def impl_ss(num):
    tmp = []
    for i, imgids in enumerate(d.imgids):
        if i < num:
            continue
        if i >= num+3:
            break
        print(i)
        x = d.load_image(imgids)
        boxes = selective_search(x, mode = MODE, random_sort = True)
        tmp.append(annotations(imgids, boxes))
    return tmp
js = OrderedDict()
nums = [3*i for i in range(3)]
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as excuter:
    result = list(excuter.map(impl_ss, nums))
js["pseudo_annotations"] = result
with open("/data/unagi0/masaoka/endoscopy/annotations/pseudo_annotations.json", "w") as f:
    json.dump(js, f, indent=2)