#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

IMG_WD=/home/aogorodnikov/data_aae
EPOCH=1500
BATCH=64
OUT=/home/aogorodnikov/aae/
latent_dim=32

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae.py -i ${IMG_WD} -e ${EPOCH} -b ${BATCH} \
  -o ${OUT} -v --input_dim 104 104 3 --latent_dim $latent_dim \
  --train --adversarial --conv

