import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 5,150)
    return canny


image = cv2.imread('test_image.jpg')
lang_image = np.copy(image)
canny = canny(lang_image)
##cv2.imshow('result',canny)
plt.imshow(canny)
cv2.waitKey(0)
