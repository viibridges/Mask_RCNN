#
# Configuration class for solar
#
import sys, os
import numpy as np
import pickle
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

    def dump(self, path):
        """dump Configuration values."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # dump to txt file
        with open(path, 'w') as fid:
            fid.writelines("Configurations:\n")
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    fid.writelines("{:30} {}\n".format(a, getattr(self, a)))

        # dump to pickle binary
        dic = {}
        pickle_path = ''.join(path.split('.txt')[:-1]) + '.pb'
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                dic[a] = getattr(self, a)
        with open(pickle_path, 'wb') as fid:
            pickle.dump(dic, fid)


# Test only
class TestConfig(SolarConfig):
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64)
    VALIDATION_STEPS = 1
    STEPS_PER_EPOCH = 2
    TRAIN_HEAD_EPOCHS = 2
    TRAIN_ALL_EPOCHS = 2


def load_config_obj(path):
    with open(path, 'rb') as fid:
        dic = pickle.load(fid)
    # convert dictionary to a config class
    obj = Config()
    for var, val in dic.items():
        setattr(obj, var, val)
    return obj


if __name__ == '__main__':
    config = SolarConfig()
    config.dump('/tmp/config.txt')

    config_loaded = load_config_obj('/tmp/config.pb')
    config_loaded.display()
