import cv2
import numpy as np
import pandas as pd
import os
import math

##############################################################################

def empty(a):
    pass
cv2.namedWindow('AREA')
cv2.resizeWindow('AREA', 1000,600)
cv2.createTrackbar('Area', 'AREA', 0, 100000, empty)

##############################################################################

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 1,1)

        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blr = cv2.GaussianBlur(img_gry, (9,9), 9)
        img_cny = cv2.Canny(img_blr, 20,20)
        kernel = np.ones((15,15))
        img_dlt = cv2.dilate(img_cny, kernel, iterations=2)
        img_erd = cv2.erode(img_dlt, kernel, iterations=1)

##############################################################################

        area1 = cv2.getTrackbarPos('Area', 'AREA',)

        contours, hierarchy = cv2.findContours(img_erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area1:
                points = cv2.boundingRect(contour)
                (x,y,w,h) = points
                # cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5, 2)
                cv2.drawContours(img, [contour], -1, (0,0,255), 5)

##############################################################################

                angle = cv2.minAreaRect(contour)
                box = cv2.boxPoints(angle)
                box = box.astype('int')
                p1, p2, p3, p4 = box
                (x1,y1), (x2,y2) = (p1[0], p1[1]), (p2[0], p2[1])
                (x3,y3), (x4,y4) = (p3[0], p3[1]), (p4[0], p4[1])
                cv2.polylines(img, [box], isClosed=True, color = (0,255,0), thickness = 5)

                cv2.arrowedLine(img, (x2,y2), (x3,y3), (255,0,0), 5, 2)
                len = math.sqrt((x3-x2)**2 + (y3-y2)**2)
                cv2.putText(img, f'{int(len)}', ((x2+x3)//2, (y2+y3)//2), cv2.FONT_HERSHEY_COMPLEX, 4, (255,0,255), 2)

##############################################################################

    cv2.imshow('AREA', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()