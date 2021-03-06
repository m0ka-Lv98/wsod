from pycocotools.cocoeval import COCOeval
import json
import torch
import transform as transf
from torchvision.transforms import Compose
import yaml
from dataset import MedicalBboxDataset
from model import ResNet50

def evaluate_coco_weak(val, model, model_path, save_path, aug=False, threshold=0.05):
    config = yaml.safe_load(open('./config.yaml'))
    dataset_means = json.load(open(config['dataset']['mean_file']))
    dataset_all = MedicalBboxDataset(
        config['dataset']['annotation_file'],
        config['dataset']['image_root'])
    if 'class_integration' in config['dataset']:
        dataset_all = dataset_all.integrate_classes(
            config['dataset']['class_integration']['new'],
            config['dataset']['class_integration']['map'])
    
    transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])

    dataset = dataset_all.split(val, config['dataset']['split_file'])
    dataset.set_transform(transform)
    
    
    #model = ResNet50()
    #model.load_state_dict(torch.load(f"/data/unagi0/masaoka/resnet50_classify_rotate_flip_shear[{val}].pt"))
    model = eval(model)
    model.load_state_dict(torch.load(model_path))

    results = []
    image_ids = []
    for index in range(len(dataset)):
        data = dataset[index]
        scale = 1   #data['scale']
        data['img'] = torch.from_numpy(data['img']) #.permute(2, 0, 1)
        # run network
        scores, labels, boxes = model(data['img'].unsqueeze(0).cuda().float(), e=True, aug=aug)
        
        scores = scores.cpu()
        labels = labels.cpu()
        boxes  = boxes.cpu()

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
    #json.dump(results, open(f'/data/unagi0/masaoka/retinanet/bbox_results_resnet50caug{val}.json', 'w'), indent=4)
    json.dump(results, open(save_path, 'w'), indent=4)
    # load results in COCO evaluation tool
    coco_true = dataset.coco
    #coco_pred = coco_true.loadRes(f'/data/unagi0/masaoka/retinanet/bbox_results_resnet50caug{val}.json')
    coco_pred = coco_true.loadRes(save_path)
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return

if __name__ == "__main__":
    val = [4]
    evaluate_coco_weak(val, model = "ResNet50()", model_path = f'/data/unagi0/masaoka/resnet50_classify{val}_epoch0_it500.pt',
                        save_path = f"/data/unagi0/masaoka/resnet50_v{val}.json", aug = False)