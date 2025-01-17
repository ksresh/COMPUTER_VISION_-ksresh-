import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4, 250)

def empty(a):
    pass

cv2.namedWindow('HSV')
cv2.resizeWindow('HSV', 500,250)
cv2.createTrackbar('min_hue', 'HSV', 0, 179, empty)
cv2.createTrackbar('max_hue', 'HSV', 179, 179, empty)
cv2.createTrackbar('min_sat', 'HSV', 0, 255, empty)
cv2.createTrackbar('max_sat', 'HSV', 255, 255, empty)
cv2.createTrackbar('min_val', 'HSV', 0, 255, empty)
cv2.createTrackbar('max_val', 'HSV', 255, 255, empty)


while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue_min = cv2.getTrackbarPos('min_hue', 'HSV')
        hue_max = cv2.getTrackbarPos('max_hue', 'HSV')
        sat_min = cv2.getTrackbarPos('min_sat', 'HSV')
        sat_max = cv2.getTrackbarPos('max_sat', 'HSV')
        val_min = cv2.getTrackbarPos('min_val', 'HSV')
        val_max = cv2.getTrackbarPos('max_val', 'HSV')

        min = np.array([hue_min, sat_min, val_min])
        max = np.array([hue_max, sat_max, val_max])

        mask = cv2.inRange(hsv, min, max)
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(img, img, mask=mask)

        vstack1 = np.vstack((img, hsv))
        vstack2 = np.vstack((mask_3d, result))
        hstack1 = np.hstack((vstack1, vstack2))

        #cv2.imshow('img', img)
        #cv2.imshow('hsv', hsv)
        #cv2.imshow('mask', mask)
        #cv2.imshow('result', result)
        cv2.imshow('final', hstack1)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()