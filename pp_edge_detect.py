import cv2

'''
   function to apply laplacian of gaussian algorithm for edge detection

    Arguments:
   -----------
       x: 2D np.ndarray
           list of 2D arrays for one channel (called red, but it's arbitrary)

       ddepth: cvtype value
           format of an image as openCV parameter
           Default: cv2.CV_16U

       kernel_size: int
           kernel size for laplacian transform
           Default: 3


   Returns:
   -----------
       List of 2D np.ndarray's each corresponding to a single channel, with LoG applied

'''

def pp_edge_detect(x, ddepth=cv2.CV_16U, kernel_size=3):
    out=[]
    for i in range(len(x)):
        x_gauss = cv2.GaussianBlur(x[i], (3, 3), 0)  # Gaussian denoising
        x_laplace = cv2.Laplacian(x_gauss, ddepth, kernel_size)
        x_abs_dst = cv2.convertScaleAbs(x_laplace)
        x_diff = x[i]/(x[i].max()/255.0) - x_abs_dst
        out.append(x_diff)
    return out

