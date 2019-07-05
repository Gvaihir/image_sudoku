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



# custom
from singleCell_export import slice_export


np.random.seed(6)

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
    description='''A module uses centroids and crops out individual cells with fixed crop size''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Crop wisely''')
parser.add_argument('--img_wd', default = None, help='directory with images. Default - NONE')
parser.add_argument('--meta_wd', default = None, help='directory with meta data. Default - NONE')
parser.add_argument('--crop_size', default = 150, type=int, help='Size of a cropped image. Default = 150')
parser.add_argument('--pt', default = 1, type=int, help='Plate number. Default - 1')
parser.add_argument('--rnd', default = False, type=bool, help='Select subset of images per well? Default - FALSE')
parser.add_argument('--rnd_numb', default = 10, type=int, help='Number of images to select from well. Use with --rnd=True'
                                                               ' Default - 10')
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

    # Organize data
    # import meta data
    X_names = pd.DataFrame(sorted(glob(os.path.join(argsP.img_wd, '*.tif*'))))
    X_names['plate'] = X_names.loc[:, 0].str.extract(r'(Pt\d+)')
    X_names['well'] = X_names.loc[:, 0].str.extract(r'(r\d+c\d+)')
    X_names['field'] = X_names.loc[:, 0].str.extract(r'(f\d+)')
    X_names['file_name'] = X_names.loc[:, 0].str.extract(r'(r\d+c\d+f\d+)')

    # create output dir
    if not os.path.exists(argsP.out):
        os.makedirs(argsP.out)

    # random selection?
    if argsP.rnd:
        X_names = X_names.groupby('well').apply(lambda x: x.sample(argsP.rnd_numb)).reset_index(drop=True)

    for i in range(X_names.shape[0]):

        # import image
        X = cv2.imread(X_names.loc[0, i], -1)

        # import meta data in JSON
        json_fileName = ".".join([X_names.file_name[i], "json"])
        with open(os.path.join(argsP.meta_wd, json_fileName)) as json_file:
            data = json.load(json_file)

        for j in range(len(data["points"])):
            crop_img = slice_export(img=X, points=data["points"][j], size=argsP.crop_size)
            if crop_img.size == 0:
                continue

            # export
            cv2.imwrite(os.path.join(argsP.out, "_".join(['Pt{0:02d}'.format(argsP.pt),
                                                          X_names['base'][i],
                                                          X_names['field'][i],
                                                          '{0:04d}.tif'.format(j)])), crop_img)





