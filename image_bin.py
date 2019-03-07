# splitting and CLAHE correction function
def split_func(x, tile):
    # exclusion of pixels on the edges for even split of the image
    dim_0_index = list(range(int((x.shape[0] - x.shape[0] // tile * tile) / 2),
                             int((x.shape[0] + x.shape[0] // tile * tile) / 2)))

    dim_1_index = list(range(int((x.shape[1] - x.shape[1] // tile * tile) / 2),
                             int((x.shape[1] + x.shape[1] // tile * tile) / 2)))

    # delete edges
    x_rm_edges = x[dim_0_index,:][:,dim_1_index]

    # new dimensions
    x_dim0, x_dim1 = x_rm_edges.shape[0], x_rm_edges.shape[1]

    # number of equal parts
    #parts0, parts1 = x_dim0//tile, x_dim1//tile


    # create new list for all tiles
    list_tiles = []

    # split into equal tiles
    for j in range(0, x_dim0, tile):
        for k in range(0, x_dim1, tile):
            j1 = j + tile
            k1 = k + tile
            tiles = x_rm_edges[j:j1,k:k1]
            cl_tiles = clahe.apply(tiles)
            list_tiles.append(cl_tiles)

    return list_tiles





