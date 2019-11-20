# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

from __future__ import print_function

try:
    raw_input
except:
    raw_input = input

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import sys


# plotting and other
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from absl import flags

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
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()

'''
1. read in JSON
2. get row name, create subdir if doesnt exist 
3. create sym links for anomalies 
4. create grid of examples of anomalies and normal per row
'''


# MAIN FUNC
if __name__ == "__main__":




