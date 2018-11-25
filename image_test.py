"""testing basic image processing functions"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir

def channel_merge(x, b, b_coeff, g, g_coeff, r, r_coeff, workDir):
    # adaptive equalizing
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(b)
    cl2 = clahe.apply(g)
    cl3 = clahe.apply(r)

    # combine channels
    img = np.zeros((b.shape[0], b.shape[1], 3))
    img[:, :, 0] = cl1 / b_coeff
    img[:, :, 1] = cl2 / g_coeff
    img[:, :, 2] = cl3 / r_coeff

    # plot histograms for every channel
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([lol], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

    # write image
    cv2.imwrite("/".join([workDir, x]), img)


# import images
b = cv2.imread("/Users/antonogorodnikov/Desktop/img/r01c01f01p01-ch1sk1fk1fl1.tiff", -1)
g = cv2.imread("/Users/antonogorodnikov/Desktop/img/r01c01f01p01-ch2sk1fk1fl1.tiff", -1)
r = cv2.imread("/Users/antonogorodnikov/Desktop/img/r01c01f01p01-ch3sk1fk1fl1.tiff", -1)


20, 100, 50

direct = listdir("/Users/antonogorodnikov/Desktop/img")
for x in