import numpy as np
import os
import json
import random
import sys
import abc
import skimage.io, skimage.color

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils, visualize

def adapt_mvp_path(path):
    if os.path.exists(path):
        return path
    else:
        # insert nfs/ next to ~/
        path = os.path.expanduser(path)
        home_path = os.path.expanduser('~')
        return path.replace(home_path, '/workspace/nfs', 1)
  

def get_dataset(dataset, mode):
    """
    Get dataset object by its name and mode (train/test)
    """
    if None: pass 
    elif dataset == 'simple':
        ds = SolarDatasetSimple()
        ds.initialize(mode)
    else:
        raise ValueError("No dataset {} with mode {} found.".format(dataset, mode))
    
    return ds

class SolarDatasetBase(utils.Dataset):
    """Solar panel dataset base class
    """
    @abc.abstractmethod
    def initialize(self, mode='train'):
        """Initialize the dataset, load paths to images and corresponding annotations 
        """
        assert False, "This is an abstract method"
        
        
    def _prepare_data(self, data_dir, split_file):
    
        # Add images
        # read the given split file, load list of image paths and annotation paths
        with open(split_file, 'r') as fid:
            idx = 0
            for line in fid.readlines():
                img_file, ann_file = line.strip().split(',')
                ann_path = os.path.join(data_dir, ann_file)
                img_path = os.path.join(data_dir, img_file)
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    ann_dict = json.load(open(ann_path, 'r'))
                    box_list = ann_dict['shapes']
                    if len(box_list) == 0:
                        # print('Img %s box_list is 0'%img_path)
                        continue
                    bboxes = []
                    labels = []
                    for item_box in box_list:
                        label = item_box['label']
                        points = item_box['points']
                        if len(points) != 2:
                            print('Annotation of img %s is abnormal.'%img_file)
                            break  # Do not use 'continue' here. Drop the img if one annotation is wrong
                        x_min = np.minimum(points[0][0], points[1][0])
                        y_min = np.minimum(points[0][1], points[1][1])
                        x_max = np.maximum(points[0][0], points[1][0])
                        y_max = np.maximum(points[0][1], points[1][1])

                        bboxes.append((x_min, y_min, x_max, y_max))
                        labels.append(label)

                    self.add_image("shapes", image_id=idx, path=img_path,
                                  bboxes=bboxes, labels=labels)
                    idx += 1

                    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        if image.ndim == 2:  # if gray image
            image = skimage.color.gray2rgb(image)
        assert(image.ndim == 3)
        self.image_info[image_id].update({'height': image.shape[0], 'width': image.shape[1]})
        return image

    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "solar":
            return info["solar panel defects diagnosis"]
        else:
            super(self.__class__).image_reference(self, image_id)

            
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        count = len(info['bboxes'])
        if 'height' not in info:
            image = skimage.io.imread(info['path'])
            self.image_info[image_id].update({'height': image.shape[0], 'width': image.shape[1]})
            info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'] ,count], dtype=np.uint8)
        for i, bbox in enumerate(info['bboxes']):
            x_min, y_min, x_max, y_max = [int(x) for x in bbox]
            mask[y_min:y_max+1, x_min:x_max+1, i] = 1
        class_ids = np.array([int(l)+2 for l in info['labels']])
        return mask.astype(np.bool), class_ids.astype(np.int32)
    
    
    
    
class SolarDatasetSimple(SolarDatasetBase):
    """Simple Solar panel dataset
    """
    def initialize(self, mode='train'):
        """Initialize the dataset, load paths to images and corresponding annotations 
        """
        # Add classes
        self.add_class("shapes", 1, "unknown")
        self.add_class("shapes", 2, "type0")
        self.add_class("shapes", 3, "type1")
        self.add_class("shapes", 4, "type2")
        
        # prepare data
        root_dir = os.path.expanduser("~/data/solar_panel/")
        root_dir = adapt_mvp_path(root_dir)
        data_dir = os.path.join(root_dir, "images/20190322/")
        split_file  = os.path.join(root_dir, "splits/20190322/{}.csv".format(mode))
        self._prepare_data(data_dir, split_file)
