import numpy as np
import cv2

def mse(img_1, img_2):
   h, w = img_1.shape
   diff = cv2.subtract(img_1, img_2)
   err = np.sum(diff**2)
   mse = err/float(h*w)
   return mse