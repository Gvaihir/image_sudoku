# conda
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np

def slice_export(img, coord, size, coef_dist=0.5):

    # detect top left corner of image
    start_coord=np.array(coord)-int(size*coef_dist)

    # crop
    img_crop=img[start_coord[0]:start_coord[0]+size, start_coord[1]:start_coord[1]+size]

    return img_crop
