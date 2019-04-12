# conda
from __future__ import print_function, unicode_literals, absolute_import, division

import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import cv2


# pip
from csbdeep.utils import Path, normalize
from skimage.segmentation import find_boundaries
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, draw_polygons, sample_points
from stardist import Config, StarDist
from starDist_predict import stardist_predict


def slice_export(img, coord, size, coef_dist=0.5):

    # detect top left corner of image
    start_coord=np.array(coord)-int(size*coef_dist)
    img_crop=img[start_coord[0]:start_coord[0]+size, start_coord[1]:start_coord[1]+size]

    return img_crop
