
    # exclusion of pixels on the edges for even split of the image
    dim_0_index = list(range(int((x.shape[0] - x.shape[0] // tile * tile) / 2),
                             int((x.shape[0] + x.shape[0] // tile * tile) / 2)))

    dim_1_index = list(range(int((x.shape[1] - x.shape[1] // tile * tile) / 2),
                             int((x.shape[1] + x.shape[1] // tile * tile) / 2)))

    # delete edges
    x_rm_edges = x[dim_0_index,:][:,dim_1_index]

    # adaptive equalizing
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(b)
    cl2 = clahe.apply(g)
    cl3 = clahe.apply(r)



