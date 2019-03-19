import cv2


# splitting and CLAHE correction function
def split_func(x, ntiles, tile_size, clahe):

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
