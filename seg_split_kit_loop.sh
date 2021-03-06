# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

test=/sudoku/kaggle_data/train

for dir in $test/*;
do
for subdir in $dir/*;
do

base_dir=$(basename -- "$dir")
base_sub=$(basename -- "$subdir")

WD=$subdir
MODEL="/home/aogorodnikov/models"
PT=$(echo $subdir | rev | cut -b1 | rev)
OUT=/sudoku/meta_train/$base_dir/$base_sub
DEV="2,3"
FORMAT="png"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p ${OUT}

CUDA_VISIBLE_DEVICES=${DEV} ${PYTHON} seg_split_kit.py --wd ${WD} --model ${MODEL} --pt ${PT} \
    --format=${FORMAT} --out ${OUT}

done
done







