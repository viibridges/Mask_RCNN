#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2

import json
import skimage.io, skimage.transform, skimage.color

import warnings
warnings.filterwarnings('ignore')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.abspath("logs")


#
# Load configuration
#
from config_solar import SolarConfig, SMALL_CONFIG
config = SolarConfig()
config.display()


#
# Load training/testing data
#
from dataset_solar import get_dataset
dataset_name = '20190401'

# Training dataset
dataset_train = get_dataset(dataset_name, 'train')
dataset_train.prepare()

# Validation dataset
dataset_val = get_dataset(dataset_name, 'val')
dataset_val.prepare()


#
# ## Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


#
# Which weights to start with?
#
init_with = "coco"  # imagenet, coco, or last

if init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    assert(os.path.exists(COCO_MODEL_PATH))
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# define augmenter for training
from imgaug import augmenters as iaa

augmenters = iaa.Sequential([
    iaa.Fliplr(.5),
    iaa.Flipud(.5),
    iaa.Crop((10,10), keep_size=True)  # jittering
])


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            augmentation=augmenters,
            epochs=2 if SMALL_CONFIG else 200, 
            layers='heads')


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "headonly.h5")
model.keras_model.save_weights(model_path)


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            augmentation=augmenters,
            epochs=2 if SMALL_CONFIG else 200, 
            layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "complete.h5")
model.keras_model.save_weights(model_path)
