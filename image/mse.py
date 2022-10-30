import numpy as np
import cv2

def mse(img_1, img_2):
   """
   Calculates Mean Squared Error between two grayscale images with the same shape.

   Args:
      img_1 (cv2.Mat): image to calculate MSE
      img_2 (cv2.Mat): image to calculate MSE

   Returns:
      float: MSE of the two images.
   """
   
   h, w = img_1.shape
   diff = cv2.subtract(img_1, img_2)
   err = np.sum(diff**2)
   mse = err/float(h*w)
   return mse