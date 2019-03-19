import cv2
import numpy as np
import os
import pandas as pd
import glob
import sys
import argparse
from argparse import RawTextHelpFormatter
from image_bin import split_func
import pp_combine_export


parser = argparse.ArgumentParser(
    description='''Program to merge fluorescent image channels and prepare for segmentation''',
    formatter_class=RawTextHelpFormatter,
    epilog="""Merge wisely""")
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--b_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * b_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--g_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * g_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--r_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * r_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--tile_size', default = 256, type=int, help='INT Size of a tile for splitting')
parser.add_argument('--ntiles', default = 4, type=int, help='INT Number of tiles to split into. Default = 4 (each image will be split into 4)')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



# merge function
def channels_array(b, g, r):

    # combine channels
    img = np.zeros((b.shape[0], b.shape[1], 3))
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img


# ignoring hidden files
def listdir_nohidden(path):
    return [os.path.basename(x) for x in glob.glob(os.path.join(path, '*'))]


if __name__ == "__main__":
    inPath = argsP.wd
    inDir = listdir_nohidden(inPath)
    inDir.sort()
    dfDir = pd.DataFrame(inDir, columns=['Name'])
    outPath = '/'.join([os.path.dirname(inPath), "converted"])
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # make clahe function
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

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

        # work on different channels
        # perform split and correct function
        b_split_clahe = split_func(x=b, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)
        g_split_clahe = split_func(x=g, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)
        r_split_clahe = split_func(x=r, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)

        pp_combine_export(b_list=b_split_clahe, g_list=g_split_clahe, r_list=r_split_clahe,
                          b_coeff=argsP.b_coeff, g_coeff=argsP.g_coeff, r_coeff=r_coeff,
                          wdir=outPath, imName=x)

