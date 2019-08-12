# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

cwd="/sudoku/train"
CROP=90
for cell_line in $cwd/*;
do
  cl=$(basename -- "$cell_line")
  for plate in $cwd/$cl/*;
  do
    pt=$(basename -- "$plate")
    p=${pt: -1}
    IMG=$cwd/$cl/$pt/converted_rgb
    META=/sudoku/meta_train/$cl/$pt
    PLATE=$p
    OUT=/sudoku/crop_rgb/train/$cl/$pt

    PYTHON="/home/aogorodnikov/anaconda3/envs/imgsudoku/bin/python"
    mkdir -p ${OUT}
    ${PYTHON} crop_module.py --img_wd ${IMG} --meta_wd ${META} --pt ${PLATE} \
        --example True --example_prob 0.02 --out ${OUT} --crop_size 90 --format "png"

  done
done