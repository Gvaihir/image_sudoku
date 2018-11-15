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


img = cv2.merge((cl1, cl2, cl3))


plt.hist(img.ravel(),256,[0,256]); plt.show()





plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


cv2.imwrite('/Users/ogorodnikov/Desktop/clahe_2.tiff',img)


equ = cv2.equalizeHist(b)
res = np.hstack((b,equ)) #stacking images side-by-side
cv2.imwrite('/Users/ogorodnikov/Desktop/res.tiff',b)

