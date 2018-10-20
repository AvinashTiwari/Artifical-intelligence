import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 5,150)
    return canny

def display_lines(images,lines):
    line_image = np.zeros_like(images)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1), (x2,y2),(255,0,0),10)
    return  line_image


def region_of_interset(image):
    height = image.shape[0]
    polygon = np.array([[(200,height),(1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image;


image = cv2.imread('test_image.jpg')
lang_image = np.copy(image)
canny = canny(lang_image)
cropped_image = region_of_interset(canny)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40, maxLineGap=5)
line_image = display_lines(lang_image,lines)
combo_image = cv2.addWeighted(lang_image, 0.8, line_image, 1, 1)
cv2.imshow('result',combo_image)
#plt.imshow(canny)
#plt.show()
cv2.waitKey(0)
