#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

IMG_WD=/home/aogorodnikov/data_aae
EPOCH=1500
BATCH=128
OUT=/home/aogorodnikov/alexxela/
latent_dim=128

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae_alexxela.py -i ${IMG_WD} -e ${EPOCH} -b ${BATCH} \
  -o ${OUT} -v --latent_dim $latent_dim \
  --train --adversarial --conv --sobel

