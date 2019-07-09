# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

pt="Pt11"
row="r02"

IMG="/sudoku/screen_converted_rgb/"$pt/$row
META="/sudoku/meta/"$pt/$row
PLATE="11"
OUT="/sudoku/crop_rgb/"$pt/$row

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p ${OUT}

${PYTHON} crop_module.py --img_wd ${IMG} --meta_wd ${META} --pt ${PLATE} \
        --example True --example_prob 0.01 --out ${OUT}
