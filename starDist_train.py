# conda
from __future__ import print_function, unicode_literals, absolute_import, division

import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob

# pip
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap
from stardist import Config, StarDist

np.random.seed(42)
lbl_cmap = random_label_cmap()


# arguments
parser = argparse.ArgumentParser(
    description='''Training on segmented image data using StarDist''', formatter_class=RawTextHelpFormatter,
    epilog='''Train wisely''')
parser.add_argument('--img', default=os.path.join(os.getcwd(), 'img'), help='directory with images. Default - WD/img')
parser.add_argument('--mask', default=os.path.join(os.getcwd(), 'mask'), help='directory with masks. Default - WD/mask')
parser.add_argument('--shape_completion', default=False, type=bool, help='complete shapes of cells on the border. Default - False')
parser.add_argument('--testSplit', default=0.15, type=float, help='proportion of test data. Default - 0.15')
#parser.add_argument('-o', default=os.path.join(os.getcwd(), 'models'), help='output dir. Default - WD/models')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()

if __name__ == "__main__":

    # store separate variables for argparse arguments
    argsP_img = argsP.img
    argsP_mask = argsP.mask
    argsP_testSplit = argsP.testSplit
    argsP_shape_completion = argsP.shape_completion

    # features and responses
    X = sorted(glob(os.path.join(argsP_img, '*.tif')))
    Y = sorted(glob(os.path.join(argsP_mask, '*.tif')))
    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

    # read images
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    # Normalize images and fill small label holes
    axis_norm = (0, 1)  # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print(f"Normalizing image channels {'jointly' if axis_norm is None or 2 in axis_norm else 'independently'}.")
        sys.stdout.flush()

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    # split to train/validation
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = int(round(argsP_testSplit * len(X)))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    # print config object
    print(Config.__doc__)

    # train
    conf = Config(n_channel_in=n_channel, train_batch_size=4, train_shape_completion=argsP.shape_completion)
    print(conf)
    vars(conf)
    if argsP_shape_completion == False:
        model = StarDist(conf, name='stardist_no_shape_completion', basedir='models')

    else:
        model = StarDist(conf, name='stardist_shape_completion', basedir='models')

    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val))






