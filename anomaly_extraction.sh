#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

JSON=/home/aogorodnikov/data_aae
OUT=/home/aogorodnikov/anomaly_links/

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

${PYTHON} anomaly_extraction.py -j ${JSON} -o ${OUT} -a


# Required parameters

# Other parameters
parser.add_argument('-a', '--ae_loss', default=0.00, type=float, help='Reconstruction error threshold. Default - 0.00')
parser.add_argument('-d', '--adv_loss', default=0.00, type=float, help='Adversarial error threshold. Default - 0.00')
parser.add_argument('--example', action='store_true', help='Prepare a grid of images that fall to Normal and Anomaly,'
                                                           'according to thresholds')
parser.add_argument('--and_logic', action='store_true', help='Thresholds are combined in AND logic. Default = OR logic')
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')
