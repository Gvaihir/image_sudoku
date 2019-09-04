#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

img_wd=/home/aogorodnikov/data_aae
epoch=1500
batch=64
out=/home/aogorodnikov/aae/
latent_dim=32

mkdir -p $out

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae.py -i $img_wd -e $epoch -b $batch \
  -o $out -v --input_dim 104 104 3 $latent_dim \
  --train --adversarial --conv
