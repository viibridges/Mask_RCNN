#
# Configuration class for solar
#
import sys
import numpy as np
sys.path.append('../../')  # To find local version of the library

from mrcnn.config import Config

class SolarConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Solar"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 10

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 64, 128, 256, 448)  # anchor side in pixels

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.1, 1.5, 5] 

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 120

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    # Interval (number of epochs) between checkpoints 
    CHECKPOINT_EPOCH_INTERVAL = 100  # disable it and save the weights in the end

    # Only save the graph weights that achieve the best results in validation set
    # This value will override CHECKPOINT_EPOCH_INTERVAL
    SAVE_BEST_ONLY=False 
    
    # Faster rcnn mode
    FASTER_RCNN_MODE = True

    # Number of epochs for head network training
    TRAIN_HEAD_EPOCHS = 100

    # Number of epochs for all network training
    TRAIN_ALL_EPOCHS = 100

    # Image mean (RGB)
    MEAN_PIXEL = np.array([169.2, 169.2, 169.2])


    def __init__(self):
        super(SolarConfig, self).__init__()
        if self.SAVE_BEST_ONLY:
            self.CHECKPOINT_EPOCH_INTERVAL = 1000000000


# Test only
class TestConfig(SolarConfig):
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64)
    VALIDATION_STEPS = 1
    STEPS_PER_EPOCH = 2
    TRAIN_HEAD_EPOCHS = 2
    TRAIN_ALL_EPOCHS = 2


if __name__ == '__main__':
    config = SolarConfig()
    config.display()
