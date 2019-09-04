#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

IMG_WD=/home/aogorodnikov/data_aae
EPOCH=1500
BATCH=64
OUT=/home/aogorodnikov/aae/
latent_dim=32

mkdir -p $out

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae.py -i ${IMG_WD} -e ${EPOCH} -b ${BATCH} \
  -o ${OUT} -v --input_dim 144 144 3 --latent_dim $latent_dim \
  --train --adversarial --conv

