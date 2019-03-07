import cv2
import numpy as np
import os
import pandas as pd
import glob
import sys
import argparse
from argparse import RawTextHelpFormatter


parser = argparse.ArgumentParser(
    description='''Program to merge fluorescent image channels and prepare for segmentation''',
    formatter_class=RawTextHelpFormatter,
    epilog="""Merge wisely""")
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--b_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * b_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--g_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * g_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--r_coeff', default = 1, type=float, help='FLOAT Pixel intensity decrease: PI\' = PI * r_coeff / Mean(channel intensity) (Default: 1)')
parser.add_argument('--tile', default = 256, type=int, help='INT Size of a tile for splitting')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



# splitting and CLAHE correction function
def split_func(x, tile):
    # exclusion of pixels on the edges for even split of the image
    dim_0_index = list(range(int((x.shape[0] - x.shape[0] // tile * tile) / 2),
                             int((x.shape[0] + x.shape[0] // tile * tile) / 2)))

    dim_1_index = list(range(int((x.shape[1] - x.shape[1] // tile * tile) / 2),
                             int((x.shape[1] + x.shape[1] // tile * tile) / 2)))

    # delete edges
    x_rm_edges = x[dim_0_index, :][:, dim_1_index]

    # new dimensions
    x_dim0, x_dim1 = x_rm_edges.shape[0], x_rm_edges.shape[1]

    # create new list for all tiles
    list_tiles = []

    # split into equal tiles
    for j in range(0, x_dim0, tile):
        for k in range(0, x_dim1, tile):
            j1 = j + tile
            k1 = k + tile
            tiles = x_rm_edges[j:j1, k:k1]
            cl_tiles = clahe.apply(tiles)
            list_tiles.append(cl_tiles)

    return list_tiles






# merge function
def channel_merge(x, b, b_coeff, g, g_coeff, r, r_coeff, workDir):

    # adaptive equalizing
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(b)
    cl2 = clahe.apply(g)
    cl3 = clahe.apply(r)

    # combine channels
    img = np.zeros((b.shape[0], b.shape[1], 3))
    img[:, :, 0] = cl1 * b_coeff / np.mean(np.ma.masked_where(b < 255, b))
    img[:, :, 1] = cl2 * g_coeff / np.mean(np.ma.masked_where(g < 255, g))
    img[:, :, 2] = cl3 * r_coeff / np.mean(np.ma.masked_where(r < 255, r))

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



        # perform merge func
        img = channel_merge(x=x, b=b, b_coeff=argsP.b_coeff, g=g, g_coeff=argsP.g_coeff, r=r, r_coeff=argsP.r_coeff,
                      workDir=argsP.wd)

        # write image
        cv2.imwrite("".join(["/".join([outPath, x]), ".tif"]), img)
