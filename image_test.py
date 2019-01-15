"""testing basic image processing functions


x - sample
b - channel in blue
b_coeff -
g - channel in green
g_coeff
r - channel in red
r_coeff
workDir - working directory
"""

import cv2
import numpy as np
import os
import pandas as pd
import glob

import argparse
from argparse import RawTextHelpFormatter


parser = argparse.ArgumentParser(
    description='''Program to merge fluorescent image channels and prepare for segmentation''',
    formatter_class=RawTextHelpFormatter,
    epilog="""Merge wisely""")
parser.add_argument('--wd', help='directory with images')
parser.add_argument('--b_coeff', help='INT how many folds to decrease blue channel intensity (20)')
parser.add_argument('--g_coeff', help='INT how many folds to decrease green channel intensity (100)')
parser.add_argument('--r_coeff', help='INT how many folds to decrease red channel intensity (50)')
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


# merge function
def channel_merge(x, b, b_coeff, g, g_coeff, r, r_coeff, workDir):

    # adaptive equalizing
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(b)
    cl2 = clahe.apply(g)
    cl3 = clahe.apply(r)

    # combine channels
    img = np.zeros((b.shape[0], b.shape[1], 3))
    img[:, :, 0] = cl1 / b_coeff
    img[:, :, 1] = cl2 / g_coeff
    img[:, :, 2] = cl3 / r_coeff

    return img


# ignoring hidden files
def listdir_nohidden(path):
    return [os.path.basename(x) for x in glob.glob(os.path.join(path, '*'))]


def main():
    globPath = "/Users/ogorodnikov/Desktop"
    inPath = '/'.join([globPath, "img"])
    inDir = listdir_nohidden(inPath)
    inDir.sort()
    dfDir = pd.DataFrame(inDir, columns=['Name'])
    outPath = '/'.join([globPath, "converted"])
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # split file name, extract sample name
    dfDir["Sample"] = dfDir["Name"].str.split('-', expand=True)[0]

    # loop over each sample
    for num, x in enumerate(dfDir["Sample"].unique()):
        # subset df for relevant sample name
        dfRel = dfDir[dfDir.Sample == x]

        # import images for each channel
        b = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains('ch1'), 'Name'].to_string(index=False)]), -1)
        g = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains('ch2'), 'Name'].to_string(index=False)]), -1)
        r = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains('ch3'), 'Name'].to_string(index=False)]), -1)

        # perform main func
        img = channel_merge(x=x, b=b, b_coeff=argsP.b_coeff, g=g, g_coeff=argsP.g_coeff, r=r, r_coeff=argsP.r_coeff,
                      workDir=argsP.wd)

        # write image
        cv2.imwrite("".join(["/".join([workDir, x]), ".tif"]), img)
