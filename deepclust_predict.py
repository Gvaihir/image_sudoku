import pickle
import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data


# pip
from stardist import StarDist


# custom
from starDist_predict import stardist_predict, PolyArea
from clust_centroid import dbscan_alg, find_medoid
from singleCell_export import slice_export

np.random.seed(6)
lbl_cmap = random_label_cmap()

# restriction to float for argument '--example_prob'

# arguments
parser = argparse.ArgumentParser(
    description='''A module to infer classes of images based on deepcluster trained net''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Infer classes wisely''')
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--model', default=os.path.join(os.getcwd(), 'models'),
                    help='directory with models. Default - WD/models')
parser.add_argument('--out', default=os.path.join(os.getcwd(), 'clust_visual'), help='output dir. Default - WD/cropped')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()




if __name__ == "__main__":
    with open(os.path.join(argsP.model, 'cluster'), "rb") as f:
        b = pickle.load(f, encoding='latin1')

    # get image names in the directory
    im_names = [os.path.basename(x) for x in sorted(glob(os.path.join(argsP.wd, '*.tif*')))]





