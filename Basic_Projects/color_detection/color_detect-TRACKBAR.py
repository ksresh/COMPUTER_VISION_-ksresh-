import cv2
import numpy as np

##################################################################

img = cv2.imread('car.png')
img = cv2.resize(img, (0,0), None, 0.5, 0.5)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

##################################################################

def empty(a):
    pass

cv2.namedWindow('TRACK')
cv2.resizeWindow('TRACK', 720, 420)
cv2.createTrackbar('hue_min', 'TRACK', 0, 300, empty)
cv2.createTrackbar('hue_max', 'TRACK', 300, 300, empty)
cv2.createTrackbar('sat_min', 'TRACK', 0, 300, empty)
cv2.createTrackbar('sat_max', 'TRACK', 300, 300, empty)
cv2.createTrackbar('val_min', 'TRACK', 0, 300, empty)
cv2.createTrackbar('val_max', 'TRACK', 00, 300, empty)


##################################################################

gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gry_co = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)

blr = cv2.GaussianBlur(gry_co, (9,9), 1)

cny = cv2.Canny(blr, 50, 50)
cny_co = cv2.cvtColor(cny, cv2.COLOR_GRAY2BGR)

kernel = np.ones((5,5))
dlt = cv2.dilate(cny_co, kernel, iterations = 2)

erd = cv2.erode(dlt, kernel, iterations=2)

##################################################################

while True:
    hu_mi = cv2.getTrackbarPos('hue_min', 'TRACK')  # creating trackbar position
    hu_ma = cv2.getTrackbarPos('hue_max', 'TRACK')
    sa_mi = cv2.getTrackbarPos('sat_min', 'TRACK')
    sa_ma = cv2.getTrackbarPos('sat_max', 'TRACK')
    va_mi = cv2.getTrackbarPos('val_min', 'TRACK')
    va_ma = cv2.getTrackbarPos('val_max', 'TRACK')

    min = np.array([hu_mi, sa_mi, va_mi])  # defining range
    max = np.array([hu_ma, sa_ma, va_ma])

    mask = cv2.inRange(hsv, min, max) # mask matrix creation
    mask_co = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    fi_img = cv2.bitwise_and(img, img, mask)  # filter image

##################################################################

    hstack = np.hstack([img, mask_co, fi_img])
    cv2.imshow('TRACK', hstack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

##################################################################