"""testing basic image processing functions"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

# import images
b = cv2.imread("/Users/ogorodnikov/Box Sync/Pheno-Sudoku/Opera Phoenix 100318/HEK293 cytokinesis test__2018-10-03T16_10_58-Measurement 1/Images/r01c01f01p01-ch1sk1fk1fl1.tiff", -1)
g = cv2.imread("/Users/ogorodnikov/Box Sync/Pheno-Sudoku/Opera Phoenix 100318/HEK293 cytokinesis test__2018-10-03T16_10_58-Measurement 1/Images/r01c01f01p01-ch2sk1fk1fl1.tiff", -1)
r = cv2.imread("/Users/ogorodnikov/Box Sync/Pheno-Sudoku/Opera Phoenix 100318/HEK293 cytokinesis test__2018-10-03T16_10_58-Measurement 1/Images/r01c01f01p01-ch3sk1fk1fl1.tiff", -1)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(b)
cl2 = clahe.apply(g)
cl3 = clahe.apply(r)

needed_multi_channel_img = np.zeros((b.shape[0], b.shape[1], 3))
needed_multi_channel_img [:,:,0] = b/5
needed_multi_channel_img [:,:,1] = g/5
needed_multi_channel_img [:,:,2] = r/5

img = cv2.merge((b, g, r))


plt.hist(img.ravel(),256,[0,256]); plt.show()





plt.imshow(needed_multi_channel_img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


cv2.imwrite('/Users/ogorodnikov/Desktop/b.tiff',needed_multi_channel_img)


equ = cv2.equalizeHist(b)
res = np.hstack((b,equ)) #stacking images side-by-side
cv2.imwrite('/Users/ogorodnikov/Desktop/res.tiff',b)

