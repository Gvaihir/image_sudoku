#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
conda activate imgSudoku

IMG_WD=/home/aogorodnikov/test
MODELS=/home/aogorodnikov/aae_5k
BATCH=16
OUT=/home/aogorodnikov/aae_filter_classes

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae_filter.py -i ${IMG_WD} -m ${MODELS} -b ${BATCH} \
  -o ${OUT} -v
