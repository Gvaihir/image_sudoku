# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

pt="Pt04"
row="r01"

img="/sudoku/screen_converted_rgb/"$pt/$row
meta="/sudoku/meta/"$pt/$row
plate="4"
out="/sudoku/crop_rgb/"$pt/$row

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p $out


${PYTHON} crop_module.py --img_wd $img --meta_wd $meta --pt $plate \
        --example True --example_prob 0.01 --out $out
