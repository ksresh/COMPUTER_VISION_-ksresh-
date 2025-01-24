import math

import cv2
import numpy as np
import pandas as pd

##############################################################################

def rotation(contour, img):
    rotate = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rotate)
    points = [tuple(point) for point in points]
    pt1, pt2, pt3, pt4 = points
    (x1,y1) = int(pt1[0]), int(pt1[1])
    (x2,y2) = int(pt2[0]), int(pt2[1])
    (x3,y3) = int(pt3[0]), int(pt3[1])
    (x4,y4) = int(pt4[0]), int(pt4[1])

    cv2.arrowedLine(img, (x1, y1), (x2, y2), (255, 0, 0), 2, 1)

    len1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    len1 = round(len1, 2)
    mid_point1 = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.putText(img, f'{len1}', mid_point1, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    return (x1,y1), (x2,y2), (x3,y3), (x4,y4)

##############################################################################

def track():
    def empty(a):
        pass

    cv2.namedWindow('AREA')
    cv2.resizeWindow('AREA', 640, 420)
    cv2.createTrackbar('area', 'AREA', 1000, 100000, empty )

    Area = cv2.getTrackbarPos('area', 'AREA')
    return Area

##############################################################################

def util(img):
    img = cv2.resize(img, (0,0), None, 1,1)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gry, (9,9), 3)
    cny = cv2.Canny(blr, 50, 50)
    kernel=np.ones((7,7))
    dlt = cv2.dilate(cny, kernel, iterations=2)
    erd = cv2.erode(dlt, kernel, iterations=2)

    Area = track()
    contours, hierarchy = cv2.findContours(erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    my_data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >  Area:
            peri = cv2.arcLength(contour, closed=True)
            edge = cv2.approxPolyDP(contour, 0.2*peri, closed=True)
            bo_box = cv2.boundingRect(contour)
            (x,y,w,h) = bo_box

            data = {'area' : area,
                    'peri' : peri,
                    'edge' : edge,
                    '(x,y,w,h)' : bo_box}
            my_data.append(data)

            cv2.drawContours(img, [contour], -1, (0,0,255), 2)
            cv2.rectangle(img, (x,y), (w+x,h+y), (0,255,0), 2, 2)
            cv2.putText(img, f'area : {area}', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
            cv2.polylines(img, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rotation(contour, img)

            # cv2.imshow('AREA1', img)
        else:
            cv2.imshow('AREA', img)

    if my_data:
        df = pd.DataFrame(my_data)
        df_sort = df.sort_values(by = 'area', ascending=False)
    else:
        df_sort = pd.DataFrame(my_data)

    return df_sort



