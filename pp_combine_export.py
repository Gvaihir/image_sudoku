import cv2
import numpy as np
import os


def pp_combine_export(b_list, g_list, r_list, b_coeff, g_coeff, r_coeff, outpath, imName):

    for i in range(0, len(b_list)):
        # make blank image
        img = np.zeros((b_list[i].shape[0], b_list[i].shape[1], 3))
        img[:, :, 0] = b_list[i]/(b_list[i].max()/255.0)*b_coeff
        img[:, :, 1] = g_list[i]/(g_list[i].max()/255.0)*g_coeff
        img[:, :, 2] = r_list[i]/(r_list[i].max()/255.0)*r_coeff

        cv2.imwrite(os.path.join(outpath, "_".join([imName, '{0:02d}.tif'.format(i)])), img)


