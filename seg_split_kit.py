# conda
from __future__ import print_function, unicode_literals, absolute_import, division

import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import cv2
import pandas as pd



# pip
from csbdeep.utils import Path, normalize
from skimage.segmentation import find_boundaries
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, draw_polygons, sample_points
from stardist import Config, StarDist
from starDist_predict import stardist_predict


# custom
from starDist_predict import stardist_predict, PolyArea
from clust_centroid import dbscan_alg, find_medoid
from singleCell_export import slice_export

np.random.seed(6)
lbl_cmap = random_label_cmap()


# arguments
parser = argparse.ArgumentParser(
    description='''A module predicts centroids, masks and crops images of certain size. There is an option to 
    skip the centroids if their coordinates closer to the previously analyzed centroid then a certain value''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Predict wisely''')
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--model', default=os.path.join(os.getcwd(), 'models'),
                    help='directory with models. Default - WD/models/stardist_no_shape_completion')
parser.add_argument('--pt', default = 1, type=int, help='Plate number. Default - 1')
parser.add_argument('--rnd', default = False, type=bool, help='Select subset of images per well? Default - FALSE')
parser.add_argument('--rnd_numb', default = 10, type=int, help='Number of images to select from well. Use with --rnd=True'
                                                               ' Default - 10')
parser.add_argument('-channels', default=os.path.join(os.getcwd(), 'predicted_masks'), help='output dir. Default - WD/predicted_masks')
parser.add_argument('-o', default=os.path.join(os.getcwd(), 'predicted_masks'), help='output dir. Default - WD/predicted_masks')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":

    # import images
    X_names = pd.DataFrame(sorted(glob(os.path.join(argsP.wd, '*.tif*'))))
    X_names['base'] = X_names.loc[:,0].str.extract(r'(r\d+c\d+)')

    # random selection?
    if argsP.rnd:
        X_names = X_names.groupby('base').apply(lambda x: x.sample(argsP.rnd_numb)).reset_index(drop=True)


    X = [cv2.imread(x, -1) for x in X_names.loc[:,0]]

    # import model
    model = StarDist(None, name='stardist_no_shape_completion', basedir=argsP.model)

    ###### PREDICT FOR EACH IMAGE ######
    for i in range(0, len(X)):
        coord, points = stardist_predict(X[i], model=model, size=72, prob_thresh=0.7, nms_thresh=0.7)

        # exclude points based on the polygon surface
        # estimate area
        area = [PolyArea(x, coord) for x in points]

        # perform filter by area
        points = [points[x] for x in range(len(points)) if area[x] > 100]

        # perform DBSCAN algorithm
        labels = dbscan_alg(points=points, eps=30, min_samples=2)

        # get individual points, which have no cluster
        points_filt = [points[x] for x in range(len(points)) if labels[x] == -1]

        # get medoid points for clusters
        point_clust = [find_medoid(x, points=points, labels=labels) for x in list(set(labels[labels != -1]))]

        # append both lists
        points_final = points_filt + point_clust

        # split and export single cells
        for j in range(0, len(points_final)):
            crop_img = slice_export(img = X[i], points=points_final[j], size = 70)

            # export
            cv2.imwrite(os.path.join(outpath, "_".join([imName, '{0:02d}.tif'.format(i)])), img)







    # remove points too close to each other









    # load models
    bname = os.path.basename(os.path.dirname(argsP.model))
    model = StarDist(None, name=argsP.model, basedir=bname)






