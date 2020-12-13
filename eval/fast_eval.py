from pycocotools.cocoeval import COCOeval
import json
import torch
from utils.anchor import make_anchor

def evaluate_coco(dataset, model, val, name, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        anch = make_anchor()

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            data['img'] = torch.from_numpy(data['img']) #.permute(2, 0, 1)
            print(data['p_bboxes'].shape,anch.shape)
            rois = torch.from_numpy(data['p_bboxes']) if data['annot'].shape!=(0,5) else anch
            n = min(rois.shape[0],2000)
            rois = rois[:n]

            # run network
            scores, labels, boxes = model(data['img'].cuda().float().unsqueeze(dim=0),0,rois.unsqueeze(0).cuda().float())
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
        json.dump(results, open(f'/data/unagi0/masaoka/wsod/result_bbox/{name}{val}.json', 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(f'/data/unagi0/masaoka/wsod/result_bbox/{name}{val}.json')

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return

if __name__=='__main__':
    val=0
    name = 'slv3'
    dataset = torch.load(f'/data/unagi0/masaoka/val_all{val}.pt')
    model = resnet18(num_classes=3, pretrained=True)
    model.cuda()
    model.load_state_dict(torch.load(f'/data/unagi0/masaoka/wsod/model/oicr/SLV_Retinaanchor1e-05_0_6.pt'))
    evaluate_coco(dataset, model, val, name, threshold=0.05)


#18 (midn+oicr+slv+retinanet rms)
#18fb
#SLV_Retinaanchormidn1e-05_0_6 (midn+slv+retinanet adam) midn_retina
