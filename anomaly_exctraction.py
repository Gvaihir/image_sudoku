# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

from __future__ import print_function

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import sys
import json
import pandas as pd
import re
import shutil

# plotting and other
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
import seaborn as sns
import cv2



parser = argparse.ArgumentParser(
    description='''Module to select images with anomalies according to aae (ACAE). Copies symlinks,
    keeps original directory structure''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Select wisely''')

# Required parameters
requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument('-i', '--img_wd', help='directory with images.'
                                                  'Should contain one level of subdirs (e.g. classes)', required=True)
requiredNamed.add_argument('-j', '--json_file', help='Full path to .json - output of aae_filter.py', required=True)
requiredNamed.add_argument('-o', '--output', help='Output directory, where input directory structure will be preserved'
                                                  'and symlinks of images will be created', required=True)

# Other parameters
parser.add_argument('-a', '--ae_loss', default=0.00, type=float, help='Reconstruction error threshold. Default - 0.00')
parser.add_argument('-d', '--adv_loss', default=0.00, type=float, help='Adversarial error threshold. Default - 0.00')
parser.add_argument('--example', action='store_true', help='Prepare a grid of images that fall to Normal and Anomaly,'
                                                           'according to thresholds')
parser.add_argument('--and_logic', action='store_true', help='Thresholds are combined in AND logic. Default = OR logic')
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


# Function for JSON import
def json_importer(json_file):
    '''
    Func to import JSON file and extract additional data
    :param json_file: path to JSON
    :return: PAndas Data Frame
    '''
    with open(json_file, 'r') as json_in:
        df = pd.DataFrame(json.load(json_in))
    df['well'] = [re.search(r'Pt\d+_r\d\dc\d\d', x)[0] for x in df.image]
    df['row'] = [re.search(r'r\d\d', x)[0] for x in df.image]
    return df

# MAIN FUNC
if __name__ == "__main__":

    np.random.seed(33)

    # DataFrame of images per plate
    df = json_importer(argsP.json_file)

    if argsP.verbose:
        print("JSON loaded")
        sys.stdout.flush()
    # filter by the threshold
    if argsP.and_logic:
        df_anomaly = df.loc[(df.ae_loss >= argsP.ae_loss) & (df.adv_loss >= argsP.adv_loss)]

    else: # default
        df_anomaly = df.loc[(df.ae_loss >= argsP.ae_loss) | (df.adv_loss >= argsP.adv_loss)]

    if argsP.verbose:
        print("Number of images post-threshold:{}".format(df_anomaly.shape[0]))
        sys.stdout.flush()

    # if normal data needed:
    if argsP.example:
        # select random rows and filter to keep only normal
        rand_row = np.random.choice(df.index, size=200, replace=False)
        rand_row = [x for x in rand_row if x not in df_anomaly.index]
        rand_row = np.random.choice(rand_row, size=100, replace=False)

        # keep 100 random normal images
        df = df.loc[rand_row,:]


    else:
        del df

    # CREATE OUTPUT DIRECTORIES
    subdirs = df_anomaly.row.unique()
    pt = os.path.basename(argsP.json_file).split('.')[0]
    for i in subdirs:
        outPath = os.path.join(argsP.output, pt, i)
        if not os.path.exists(outPath):
            os.makedirs(outPath)

    # CREATE SYMLINK OF ANOMALY IMAGES
    filenames_ls = [os.path.basename(x) for x in df_anomaly.image]
    source_path_ls = df_anomaly.image.to_list()
    dest_path_ls = [os.path.join(argsP.output, pt, df_anomaly.row.to_list()[x], filenames_ls[x]) for x in range(len(filenames_ls))]
    [os.symlink(source_path_ls[x], dest_path_ls[x]) for x in range(len(dest_path_ls))]


    # OPTIONAL export of 25 samples
    ## Anomaly
    if argsP.example:
        for out_type in ['anomaly', 'normal']:

            fig = plt.figure(figsize=(8, 8))
            columns = 5
            rows = 5
            if out_type == "anomaly":
                source_data = df_anomaly.image
            else:
                source_data = df.image
            rand_select = np.random.choice(source_data, 25, replace=False)
            for k in range(1, columns * rows + 1):
                img = cv2.imread(rand_select[k-1], -1)

                fig.add_subplot(rows, columns, k)
                plt.imshow(img)
                plt.axis('off')

            plt.savefig(fname=os.path.join(argsP.output, pt, '{}_example.pdf'.format(out_type)))












'''
4. create grid of examples of anomalies and normal per row
'''


