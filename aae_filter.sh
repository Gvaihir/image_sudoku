#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
conda activate imgSudoku

IMG_WD=/sudoku/crop_rgb/Pt04
MODELS=/home/aogorodnikov/aae
BATCH=16
OUT=/home/aogorodnikov/aae_filter_Pt04

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

cwd="/sudoku/train"
${PYTHON} aae_filter.py -i ${IMG_WD} -m ${MODELS} -b ${BATCH} \
  -o ${OUT} -v
