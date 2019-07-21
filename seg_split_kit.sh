# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

WD="/sudoku/screen_converted_rgb/Pt11/r02"
MODEL="/home/aogorodnikov/models"
PT="11"
OUT="/sudoku/meta/Pt11/r02"
DEV="0"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p ${OUT}

CUDA_VISIBLE_DEVICES=${DEV} ${PYTHON} seg_split_kit.py --wd ${WD} --model ${MODEL} --pt ${PT}


