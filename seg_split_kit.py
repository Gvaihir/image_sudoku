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


# custom
from polygon_surface import PolyArea
from starDist_predict import stardist_predict


np.random.seed(6)
lbl_cmap = random_label_cmap()


# arguments
parser = argparse.ArgumentParser(
    description='''A module predicts centroids, masks and crops images of certain size. There is an option to 
    skip the centroids if their coordinates closer to the previously analyzed centroid then a certain value''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Predict wisely''')
parser.add_argument('--img', default=os.path.join(os.getcwd(), 'img'), help='directory with images. Default - WD/img')
parser.add_argument('--model', default=os.path.join(os.getcwd(), 'models'),
                    help='directory with models. Default - WD/models/stardist_no_shape_completion')
parser.add_argument('-channels', default=os.path.join(os.getcwd(), 'predicted_masks'), help='output dir. Default - WD/predicted_masks')
parser.add_argument('-o', default=os.path.join(os.getcwd(), 'predicted_masks'), help='output dir. Default - WD/predicted_masks')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":
    X = sorted(glob(os.path.join(argsP_img, '*.tif')))
    X = list(map(cv2.imread, X))

    model = StarDist(None, name='stardist_no_shape_completion', basedir=argsP.model)
    # ##### PREDICT ######
    coord, point = stardist_predict(X, model=model, size=72, prob_thresh=0.7)

    # exclude points based on the polygon surface
    area = point

    # remove points too close to each other









    # load models
    bname = os.path.basename(os.path.dirname(argsP.model))
    model = StarDist(None, name=argsP.model, basedir=bname)






