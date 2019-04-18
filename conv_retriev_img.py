# conda

import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt


p = argparse.ArgumentParser()
p.add_argument("--arg", type=restricted_float)
# arguments
parser = argparse.ArgumentParser(
    description='''A module for exporting images from convolution layers of deepclust into single pdf for each layer''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Image wisely''')
parser.add_argument('--wd', default=os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--num', default=1, choices=list(range(1,257)), help='From how many filters to export images. Default - 1')
parser.add_argument('--seed', default=1, type=int, help='Random seed. Default - 1')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


np.random.seed(argsP.seed)


# ignoring hidden files
def listdir_nohidden(path):
    return [x for x in glob(os.path.join(path, 'layer*'))]

# create and export pdf function
def exp_pdf_act(layer):
    # list images
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for num, i in enumerate(glob(os.path.join(layer, '*.tif'))):
        img = cv2.imread(i, -1)
        fig.add_subplot(rows, columns, num)
        plt.imshow(img, cmap='gray');
        plt.axis('off')
        plt.title(os.path.basename(layer))
    plt.savefig(fname=os.path.join(outPath, "_".join(['Pt{0:02d}'.format(argsP.pt),
                                                       X_names['base'][i],
                                                       '{0:04d}_example.pdf'.format(j)])))



if __name__ == "__main__":
    inPath = argsP.wd
    inDir = listdir_nohidden(inPath)
    inDir.sort()
    outPath = '/'.join([os.path.dirname(inPath), "export"])



    dfDir = pd.DataFrame(inDir, columns=['Name'])
    outPath = '/'.join([os.path.dirname(inPath), "converted_grey"])
    if not os.path.exists(outPath):
        os.makedirs(outPath)




