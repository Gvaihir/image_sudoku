# conda
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

'''

   function to crop image based on centroid coordinates

    Arguments:
   -----------
       img: 2D array 
           image as 2D array of pixels
           
        points: list
            list of xy coordinates of a centroid
        
        size: int
            size of a crop in pixels

   Returns:
   -----------
        2D array - cropped image

   '''


def slice_export(img, points, size):

    # detect top left corner of image
    start_coord = np.array(points)-int(size*0.5)

    # crop
    img_crop = img[start_coord[0]:start_coord[0]+size, start_coord[1]:start_coord[1]+size]

    return img_crop
