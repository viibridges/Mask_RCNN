#!/usr/bin/env python
# coding: utf-8
import os
import sys
from ipdb import set_trace as st
import cv2
from tqdm import tqdm
from compute_mAP import compute_ap, yx2xy
import numpy as np
from imageio import imwrite

def draw_bbox(image, class_ids, bboxes):
    # define color map
    cmap = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,255,0),
        (255,0,255),
        (0,255,255),
    ]
    
    for class_id, bbox in zip(class_ids, bboxes):
        assert class_id < len(cmap), "Please add more colors to continue"
        color = cmap[class_id]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--model', required=False, help="path to your model")
    parser.add_argument('--dataset', required=False, default='20190401')
    parser.add_argument('--dataset_mode', required=False, default='test')

    args = parser.parse_args()
    model_dir = args.model_dir
    if not hasattr(args, 'model'):
        model_path = os.path.join(model_dir, 'complete.h5')
    else:
        model_path = args.model
    dataset = args.dataset
    dataset_mode = args.dataset_mode

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../../")
    sys.path.append(ROOT_DIR)  # To find local version of the library
    import mrcnn.model as modellib

    # Load training/testing data
    from dataset_solar import get_dataset
    ds = get_dataset(dataset, dataset_mode)
    ds.prepare()

    # ## Detection

    # load model configuration
    from config_solar import SolarConfig
    class InferenceConfig(SolarConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir='/dev/null')

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    result_dir = os.path.join(model_dir,\
            'results-{}-{}'.format(dataset_mode, os.path.basename(model_path)))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # ## Evaluate mAP
    pred_classes, pred_bboxes, pred_scores = [], [], []
    gt_classes, gt_bboxes = [], []
    for image_id in tqdm(ds.image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, _ = modellib.load_image_gt(ds, inference_config,
                                   image_id, use_mini_mask=False)
        # Collect groundtruth Id
        gt_classes.append(gt_class_id)
        gt_bboxes.append(yx2xy(gt_bbox))

        # Run object detection
        results = model.detect([image], verbose=0)
        res = results[0]
        pred_classes.append(res['class_ids'])
        pred_bboxes.append(yx2xy(res['rois'])) 
        pred_scores.append(res['scores'])

        # draw bbox in image
        image_pred = image.copy()
        draw_bbox(image, gt_class_id, yx2xy(gt_bbox))
        draw_bbox(image_pred, res['class_ids'], yx2xy(res['rois']))
        image_save = np.hstack([image, image_pred])
        # save image
        image_name = os.path.basename(ds.image_info[image_id]['path'])
        save_path = os.path.join(result_dir, image_name)
        imwrite(save_path, image_save)

    APs = compute_ap(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose=True)
    with open(os.path.join(result_dir, 'mAP.txt'), 'w') as fid:
        for class_id, ap in APs.items():
            fid.writelines("class '{}' AP: {:2f}%\n".format(class_id, ap*100))
        mAP = sum(APs.values()) / len(APs)
        fid.writelines("mAP: {:2f}%".format(mAP*100))
