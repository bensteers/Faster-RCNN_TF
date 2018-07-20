# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
from datasets.pascal_voc import pascal_voc
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import pdb


class uo_images(pascal_voc):
    def __init__(self, image_set, year='0000', devkit_path=None):
        imdb.__init__(self, 'uo_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'UOImages')
        self._classes = ('__background__', 'plume', 'shadow', 'cloud', 'light', 'ambiguous')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : True,#False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

#     def _load_image_set_index(self):
#         """
#         Load the indexes listed in this dataset's image set file.
#         """
#         # Example path to image set file:
#         # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
#         image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
#                                       self._image_set + '.txt')
#         assert os.path.exists(image_set_file), \
#                 'Path does not exist: {}'.format(image_set_file)
#         with open(image_set_file) as f:
#             image_index = [x.strip() for x in f.readlines()]
#         return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return cfg.DATA_DIR#os.path.join(cfg.DATA_DIR, 'Data')
    

if __name__ == '__main__':
    d = uo_images('trainval')
    res = d.roidb
    from IPython import embed; embed()
