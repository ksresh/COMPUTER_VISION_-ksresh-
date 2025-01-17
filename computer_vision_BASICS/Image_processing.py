import cv2
import numpy as np

kernel = np.ones((5,5), dtype=np.uint8)  # (5,5)-kernel size; unsinged 8 bit integer we want
print(kernel)

img = cv2.imread('paa1.jpg')
sze_img= cv2.resize(img, (300,300))  # (width, height) = (height, width, channel)
crp_img = sze_img[0:300 ,100:200]   # [y, x] cropping image
crp_img_sze = cv2.resize(crp_img, (sze_img.shape[1], sze_img.shape[0]))  # resizing the copped image to old original size
gry_img = cv2.cvtColor(sze_img, cv2.COLOR_BGR2GRAY)
 can_img = cv2.Canny(gss_img, 50,100)   # edge detection
dil_img = cv2.dilate(can_img, kernel, iterations = 1)  # more iterations more thickness
erd_img = cv2.erode(dil_img, kernel, iterations = 1)  # thins out edges in respective directions

cv2.imshow('sze_img', sze_img)
cv2.imshow('crp_img', crp_img)
cv2.imshow('crp_img_sze', crp_img_sze)
# cv2.imshow('gry_img', gry_img)
# cv2.imshow('sze_img', sze_img)
# cv2.imshow('gss_img', gss_img)
# cv2.imshow('can_img', can_img)
# cv2.imshow('dil_img', dil_img)
# cv2.imshow('erd_img', erd_img)
cv2.waitKey(0)