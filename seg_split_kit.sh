# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
conda activate imgSudoku

WD=""
MODEL=""
PT="4"
OUT=""
DEV=""
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p ${OUT}

CUDA_VISIBLE_DEVICES=${DEV} ${PYTHON} seg_split_kit.py --wd ${WD} --model ${MODEL} --pt ${PT}


