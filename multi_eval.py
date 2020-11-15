from pycocotools.cocoeval import COCOeval
import json
import torch
from make_dloader import make_data
from model import *

def evaluate_coco(dataset, model, val,threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = 1   #data['scale']
            data['img'] = torch.from_numpy(data['img']) #.permute(2, 0, 1)
            pro = min(len(data["p_bboxes"][0]),2000)
            rois = torch.from_numpy(data["p_bboxes"][:pro]).unsqueeze(0).cuda().float()
        
            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].cuda().float().unsqueeze(dim=0), 0,rois,pro)
            else:
                scores, labels, boxes = model(data['img'].float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.imgids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.imgids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            print("error")
            return

        # write output
        json.dump(results, open(f'/data/unagi0/masaoka/wsod/result_bbox/multi{val}.json', 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(f'/data/unagi0/masaoka/wsod/result_bbox/multi{val}.json')

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return

if __name__=='__main__':
    val=2
    dataset = torch.load(f'/data/unagi0/masaoka/val_all{val}.pt')
    retinanet = multi_oicr()
    retinanet.cuda()
    retinanet.load_state_dict(torch.load(f'/data/unagi0/masaoka/wsod/model/oicr/multi_oicrrms1e-05_{val}_3.pt'))
    evaluate_coco(dataset, retinanet, val,threshold=0.05)
