import cv2
import numpy as np

##################################################################

img = cv2.imread('nature.jpg')
img = cv2.resize(img, (0,0), None, 0.5,0.5)

##################################################################

def empty(a):
    pass

cv2.namedWindow('COLOR')  # CREATING TRACKBAR
cv2.resizeWindow('COLOR', 720, 420)
cv2.createTrackbar('RED', 'COLOR', 0, 255, empty)
cv2.createTrackbar('GREEN', 'COLOR', 0, 255, empty)
cv2.createTrackbar('BLUE', 'COLOR', 0, 255, empty)

##################################################################

while True:
    red = cv2.getTrackbarPos('RED', 'COLOR')  # creating trackbar positions
    green = cv2.getTrackbarPos('GREEN', 'COLOR')
    blue = cv2.getTrackbarPos('BLUE', 'COLOR')

    min_bound = np.array([red, green, blue])  # defining range
    max_bound = np.array([255, 255, 255])

    mask = cv2.inRange(img, min_bound, max_bound) # masked image
    mask_co = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fi_img = cv2.bitwise_and(img, img, mask=mask)

##################################################################

    print(f'img : {img.shape}')
    print(f'mask : {mask_co.shape}')
    print(f'fi_img : {fi_img.shape}')

##################################################################

    hstack = np.hstack([img, mask_co, fi_img])
    cv2.imshow('COLOR', hstack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

##################################################################