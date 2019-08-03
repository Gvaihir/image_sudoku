import cv2
import numpy as np
import os


def pp_combine_export(b_list, g_list, r_list, b_coeff, g_coeff, r_coeff, outpath, imName, ext):
    '''

    function to combine and export final image in 8 bit format

     Arguments:
    -----------
        b_list, g_list, r_list: lists of 2D np.ndarray's
            list of 2D arrays (output of image_bin function), blue, green and red channels

        b_coeff, g_coeff, r_coeff: int
            coefficient of intensity for each channel

        outpath: str
            directory to save images to

        imName: str
            image name

    Returns:
    -----------
    void. Writes images within function

    '''
    for i in range(0, len(b_list)):
        # make blank image
        img = np.zeros((b_list[i].shape[0], b_list[i].shape[1], 3))
        img[:, :, 0] = b_list[i]/(b_list[i].max()/255.0)*b_coeff
        img[:, :, 1] = g_list[i]/(g_list[i].max()/255.0)*g_coeff
        img[:, :, 2] = r_list[i]/(r_list[i].max()/255.0)*r_coeff

        cv2.imwrite(os.path.join(outpath, "_".join([imName, '{0:02d}.'.format(i)+str(ext)])), img)


def pp_combine_export_grey(b_list, b_coeff, outpath, imName):
    '''

    function to combine and export final image in 8 bit format for single channel images

     Arguments:
    -----------
        b_list: lists of 2D np.ndarray's
            list of 2D arrays (output of image_bin function)

        b_coeff: int
            coefficient of intensity for the channel

        outpath: str
            directory to save images to

        imName: str
            image name

    Returns:
    -----------
    void. Writes images within function

    '''
    for i in range(0, len(b_list)):
        # make blank image
        img = b_list[i]/(b_list[i].max()/255.0)*b_coeff


        cv2.imwrite(os.path.join(outpath, "_".join([imName, '{0:02d}.tif'.format(i)])), img)