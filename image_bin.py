# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

import cv2


# splitting and CLAHE correction function
def split_func(x, ntiles, tile_size, clahe):
    '''
    function to bin image and apply adaptive histogram equalizing (CLAHE)

    Arguments:
    -----------
        x: 2D np.ndarray
            Array of pixel intensities corresponding to 1 channel

        ntiles: int
            Split the image into this number of square tiles

        tile_size: int
            Size of a tile in pixels (one measure for 2 dimensions)

        clahe:  - clahe function defined in 'main' (openCV)

    Returns:
    -----------

        List of 2D np.ndarray's each corresponding to a single channel
    '''
    x_scaled = cv2.resize(x,(ntiles*tile_size, ntiles*tile_size), interpolation = cv2.INTER_AREA)

    # dimensions
    x_dim0, x_dim1 = x_scaled.shape[0], x_scaled.shape[1]

    # create new list for all tiles
    list_tiles = []

    # split into equal tiles
    for j in range(0, x_dim0, tile_size):
        for k in range(0, x_dim1, tile_size):
            j1 = j + tile_size
            k1 = k + tile_size
            tiles = x_scaled[j:j1,k:k1]
            cl_tiles = clahe.apply(tiles)
            list_tiles.append(cl_tiles)

    return list_tiles
