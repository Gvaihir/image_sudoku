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
import matplotlib.pyplot as plt
import json



# pip
from stardist import random_label_cmap
from stardist import StarDist


# custom
from starDist_predict import stardist_predict, PolyArea
from clust_centroid import dbscan_alg, find_medoid
from singleCell_export import slice_export

np.random.seed(6)
lbl_cmap = random_label_cmap()

# restriction to float for argument '--example_prob'
def restricted_float(x):
    x = float(x)
    if x < 0.00 or x > 1.00:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

p = argparse.ArgumentParser()
p.add_argument("--arg", type=restricted_float)
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
parser.add_argument('--crop', action='store_true', help='crop and export images')
parser.add_argument('--example', default = False, type=bool, help='Export 25 random examples of cells with area and coordinates?'
                                                                  'Deafault - FALSE')
parser.add_argument('--example_prob', default = 0, type=restricted_float, help='For how many images per well get the examples?'
                                                                    'Works only with --example TRUE. Default - 0')
parser.add_argument('--out', default=os.path.join(os.getcwd(), 'cropped'), help='output dir. Default - WD/cropped')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":

    # import images
    X_names = pd.DataFrame(sorted(glob(os.path.join(argsP.wd, '*.tif*'))))
    X_names['base'] = X_names.loc[:,0].str.extract(r'(r\d+c\d+)')
    X_names['field'] = X_names.loc[:, 0].str.extract(r'(f\d+)')
    X_names['file_name'] = X_names.loc[:, 0].str.extract(r'(r\d+c\d+f\d+)')

    # create output dir
    if not os.path.exists(argsP.out):
        os.makedirs(argsP.out)

    # random selection?
    if argsP.rnd:
        X_names = X_names.groupby('base').apply(lambda x: x.sample(argsP.rnd_numb)).reset_index(drop=True)

    # image import
    X = [cv2.imread(x, -1) for x in X_names.loc[:,0]]

    # import model
    model = StarDist(None, name='stardist_no_shape_completion', basedir=argsP.model)

    ###### PREDICT FOR EACH IMAGE ######
    for i in range(0, len(X)):
        coord, points_pre = stardist_predict(X[i], model=model, size=72, prob_thresh=0.7, nms_thresh=0.7)

        # exclude points based on the polygon surface
        # estimate area
        area = [PolyArea(x, coord) for x in points_pre]

        # perform filter by area
        points = [[points_pre[x], area[x]] for x in range(len(points)) if area[x] > 100]

        if len(points) < 10:
            continue

        # perform DBSCAN algorithm
        labels = dbscan_alg(points=points, eps=30, min_samples=2)

        # get individual points, which have no cluster
        points_filt = [points[x] for x in range(len(points)) if labels[x] == -1]

        # get medoid points for clusters
        point_clust = [find_medoid(x, points=points, labels=labels) for x in list(set(labels[labels != -1]))]

        # append both lists
        points_final = points_filt + point_clust

        # export as json
        result = MetaData(im_name, points_final)
        out_file = ".".join([X_names.file_name[i], 'json'])

        ### Export JSON ###
        with open(os.path.join(argsP.out, out_file), "w") as file:
            json.dump(obj.__dict__, file)

        if argsP.crop:
            # split and export single cells
            for j in range(0, len(points_final)):
                crop_img = slice_export(img=X[i], points=points_final[j], size=70)

                # export
                cv2.imwrite(os.path.join(argsP.out, "_".join(['Pt{0:02d}'.format(argsP.pt),
                                                              X_names['base'][i],
                                                              X_names['field'][i],
                                                              '{0:04d}.tif'.format(j)])), crop_img)

        # OPTIONAL export of 25 samples
        if argsP.example:

            # draw 1 or 0 with probability argsP.example_prob
            if np.random.choice(range(0, 2), p=[1 - argsP.example_prob, argsP.example_prob]) == 1:
                fig = plt.figure(figsize=(8, 8))
                columns = 5
                rows = 5
                for k in range(1, columns * rows + 1):
                    p = points_final[np.random.choice(range(1, len(points_final)))]
                    x, y = coord[p[0], p[1], 1], coord[p[0], p[1], 0]
                    lol = PolyArea(p, coord)
                    img_crop = slice_export(img=X[i], points=p, size=70)
                    fig.add_subplot(rows, columns, k)
                    plt.text(0, 0, s=lol)
                    plt.text(8, 8, s=p, color='red')
                    plt.imshow(img_crop, cmap='gray');
                    plt.axis('off')
                plt.savefig(fname=os.path.join(argsP.wd, "_".join(['Pt{0:02d}'.format(argsP.pt),
                                                                    X_names['base'][i],
                                                                    X_names['field'][i],
                                                                    '{0:04d}_example.pdf'.format(j)])))


