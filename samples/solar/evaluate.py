#!/usr/bin/env python
# coding: utf-8

import os
import sys
from ipdb import set_trace as st

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

# Load training/testing data
from dataset_solar import get_dataset
ds = get_dataset('20190401', 'test')
ds.prepare()


# ## Detection

# load model configuration
from config_solar import SolarConfig
config = SolarConfig()
config.display()
class InferenceConfig(SolarConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir='/tmp')

# Load trained weights
model_path = "logs/solar0401/complete.h5"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# ## Evaluation
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
from tqdm import tqdm
from compute_mAP import compute_mAP
pred_classes, pred_bboxes, pred_scores = [], [], []
gt_classes, gt_bboxes = [], []
for image_id in tqdm(ds.image_ids):
    # Load image and ground truth data
    image, _, gt_class_id, gt_bbox, _ = \
       modellib.load_image_gt(ds, inference_config,
                               image_id, use_mini_mask=False)
    gt_classes.append(gt_class_id)
    gt_bboxes.append(gt_bbox)

    # Run object detection
    results = model.detect([image], verbose=0)
    res = results[0]
    pred_classes.append(res['class_ids'])
    pred_bboxes.append(res['rois']) 
    pred_scores.append(res['scores'])
    
mAP = compute_mAP(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose=True)
print("mAP: ", mAP)
