#!/bin/bash

# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.
source activate imgSudoku

JSON=/home/aogorodnikov/aae_filter_Pt04/Pt04.json
OUT=/home/aogorodnikov/anomaly_links/

mkdir -p ${OUT}

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

${PYTHON} anomaly_exctraction.py -j ${JSON} -o ${OUT} -a 0.0054373 -d 0.7413648 \
  --example -v