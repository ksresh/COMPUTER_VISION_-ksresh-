import cv2
import numpy as np
import pandas as pd
import math

##############################################################################

def trackbar():
    def empty(a):
        pass

    cv2.namedWindow('AREA')
    cv2.createTrackbar('area', 'AREA', 1000, 100000, empty)

    Area = cv2.getTrackbarPos('area', 'AREA')

    return Area


##############################################################################

def length(contour, img):
    angle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(angle)
    pt1, pt2, pt3, pt4 = points.astype(int)

    (x1, y1) = int(pt1[0]), int(pt1[1])
    (x2, y2) = int(pt2[0]), int(pt2[1])
    (x3, y3) = int(pt3[0]), int(pt3[1])
    (x4, y4) = int(pt4[0]), int(pt4[1])
    cv2.arrowedLine(img,(x1, y1), (x2, y2), (255, 255, 0), 2, 2)

    mid_points = ((x1+x2)//2, (y1+y2)//2)
    (X, Y) = mid_points
    len = math.sqrt((x2-x1)**2)+((y2-y1)**2)
    len = round(len, 2)
    cv2.putText(img, f'{len}', (X,Y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 1)

    return (x1,x2), (x2,y2), (x3,y3), (x4,y4)


##############################################################################

def cont(img):
    img = cv2.resize(img, (0,0), None, 1,1)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gry, (9,9), 3)
    cny = cv2.Canny(blr, 50,50)
    kernel=np.ones((7,7))
    dlt = cv2.dilate(cny, kernel, iterations=2)
    erd = cv2.erode(dlt, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    my_data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        Area = trackbar()
        if area > Area:
            (x1, x2), (x2, y2), (x3, y3), (x4, y4) = length(contour, img)

            dis = cv2.arcLength(contour, closed=True)
            edg = cv2.approxPolyDP(contour, 0.2*dis, closed=True)
            box = cv2.boundingRect(contour)
            (x, y, w, h) = box

            cv2.drawContours(img, [contour], -2, (0,255,0), 2)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 2)
            cv2.putText(img, f'{area}', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1,)
            cv2.imshow('AREA', img)

            data = {'area': area,
                    'dis': dis,
                    'edg': edg,
                    'box': box}
            my_data.append(data)

        else:
            cv2.imshow('AREA', img)


    df = pd.DataFrame(my_data)
    df_sort = df.sort_values(by='area', ascending=False)


    return df_sort

##################################################################
img = cv2.imread('car1.jpeg')
def CONTOURS(img):
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gry_co = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)

    blr = cv2.GaussianBlur(gry, (9,9), 9)
    blr_co = cv2.cvtColor(blr, cv2.COLOR_GRAY2BGR)

    cny = cv2.Canny(blr, 50, 50)
    cny_co = cv2.cvtColor(cny, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((9,9))
    dlt = cv2.dilate(cny, kernel, iterations=3)
    dlt_co = cv2.cvtColor(dlt, cv2.COLOR_GRAY2BGR)

    erd = cv2.erode(dlt, kernel, iterations=3)
    erd_co = cv2.cvtColor(erd, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 5)
            length = cv2.arcLength(contour, closed=True)
            edges = cv2.approxPolyDP(contour, 0.4*length, closed=True)
            box = cv2.boundingRect(contour)
            (x,y,w,h) = box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 5, 1)

            rotate = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rotate)
            pt1, pt2, pt3, pt4 = points
            (x1, y1) = (int(pt1[0]), int(pt1[1]))
            (x2, y2) = (int(pt2[0]), int(pt2[1]))
            (x3, y3) = (int(pt3[0]), int(pt3[1]))
            (x4, y4) = (int(pt4[0]), int(pt4[1]))

            print(f'{area} , {length} , {len(edges)} , {box}, {points}')


    cv2.imshow('contours', img)
    cv2.waitKey(0)
    return contours

contours = CONTOURS(img)

##################################################################

'''cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 420)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 0.8, 0.8)

        contours = CONTOURS(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


##################################################################

##################################################################
##################################################################