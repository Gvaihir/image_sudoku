# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

#!/bin/bash
source activate imgSudoku

IMG=""
META=""
PT="4"
OUT=""

PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"


mkdir -p ${OUT}

${PYTHON} seg_split_kit.py ----img_wd ${IMG} --meta_wd ${META} --pt ${PT} \
        --example True --example_prob 0.01 --out ${OUT}
