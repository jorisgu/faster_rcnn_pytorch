# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np

from .pascal_voc import pascal_voc
from .imagenet3d import imagenet3d
from .kitti import kitti
from .sunrgbd_ecmr import sunrgbd
from .sunrgbd_13 import sunrgbd_13
from .inout import inout
from .rgbdp import rgbdp
from .oneraroom import oneraroom
from .kitti_tracking import kitti_tracking
from .nthu import nthu
from .coco import coco
from .kittivoc import kittivoc


def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                        pascal_voc(split, year))


# Set up kittivoc
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kittivoc_{}'.format(split)
    # print name
    __sets[name] = (lambda split=split: kittivoc(split))

# # SUNRGBD dataset
for split in ['train', 'test']:
    for encoding in ['rgb_i_100_8bits', 'd_raw_HHA_8bits','d_raw_normal_16bits','d_raw_normal_8bits']:
        name = 'sunrgbd_{}_{}'.format(split,encoding)
        __sets[name] = (lambda split=split, encoding=encoding: sunrgbd(split,encoding))
# sunrgbd_13
for split in ['train', 'test']:
    for encoding in ['rgb_i_100_8bits', 'd_raw_HHA_8bits','d_raw_normal_16bits','d_raw_normal_8bits']:
        name = 'sunrgbd_13_{}_{}'.format(split,encoding)
        __sets[name] = (lambda split=split, encoding=encoding: sunrgbd_13(split,encoding))

# # InOut dataset
for split in ['train', 'test', 'seq0', 'seq1', 'seq2', 'seq3', 'seq01', 'seq02', 'seq12', 'jg_train', 'jg_test']:
    for encoding in ['Images', 'Depth', 'Cube', 'Jet', 'HHA']:
        name = 'inout_{}_{}'.format(split,encoding)
        __sets[name] = (lambda split=split, encoding=encoding: inout(split,encoding))

# # oneraroom dataset
for split in ['all', 'easy','average','hard', 'or_2017', 'static','sar','static_monotonous','no2017']:
    for encoding in ['rgb', 'depth_8bits']:
        name = 'oneraroom_{}_{}'.format(split,encoding)
        __sets[name] = (lambda split=split, encoding=encoding: oneraroom(split,encoding))

# # rgbdp dataset
for split in ['all','train_0','train_1','train_2','train_3','train_4','test_0','test_1','test_2','test_3','test_4']:
    for encoding in ['rgb', 'depth']:
        name = 'rgbdp_{}_{}'.format(split,encoding)
        __sets[name] = (lambda split=split, encoding=encoding: rgbdp(split,encoding))


# # KITTI dataset
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}'.format(split)
    # print name
    __sets[name] = (lambda split=split: kitti(split))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    # print name
    __sets[name] = (lambda split=split: nthu(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        # print (list_imdbs())
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
