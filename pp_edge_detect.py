import cv2

def pp_edge_detect(x, ddepth=cv2.CV_16U, kernel_size=3):
    out=[]
    for i in range(len(x)):
        x_gauss = cv2.GaussianBlur(x[i], (3, 3), 0)  # Gaussian denoising
        x_laplace = cv2.Laplacian(x_gauss, ddepth, kernel_size)
        x_abs_dst = cv2.convertScaleAbs(x_laplace)
        x_diff = x[i]/(x[i].max()/255.0) - x_abs_dst
        out.append(x_diff)
    return out

