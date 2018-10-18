import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
lang_image = np.copy(image)
gray = cv2.cvtColor(lang_image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
cv2.imshow('result',blur)
cv2.waitKey(0)
