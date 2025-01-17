import cv2
import numpy as np

##################################################################

'''img = cv2.imread('car1.jpeg')
img = cv2.resize(img, (0,0), None, 1, 1)'''

##################################################################

def empty(a):
    pass

cv2.namedWindow('area1')
cv2.resizeWindow('area1', 720, 440)
cv2.createTrackbar('area2', 'area1', 100, 10000, empty)


##################################################################

def CONTOUR(erd, img):
    contours, hierarchy = cv2.findContours(erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour, area, per, poi, box, rota, points = None, 0, 0, None, None, None, None
    AREA = cv2.getTrackbarPos('area2', 'area1')

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > AREA:
            per = cv2.arcLength(contour, closed=True)
            poi = cv2.approxPolyDP(contour, 0.2*per, closed=True)
            box = cv2.boundingRect(contour)
            (x,y,w,h) = box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 5, 1)

            rota = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rota)
            points = np.array(points, dtype=np.int32)
            p1 = int(points[0][0]), int(points[0][1])
            p2 = int(points[1][0]), int(points[1][1])
            p3 = int(points[2][0]), int(points[2][1])
            p4 = int(points[3][0]), int(points[3][1])

    return contour, area, per, poi, box, rota, points

##################################################################

def IMG(img):
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gry, (9,9), 1)
    cny = cv2.Canny(blr, 100, 100)

    kernel = np.ones((5,5))
    dlt = cv2.dilate(cny, kernel, iterations=2)
    erd = cv2.erode(dlt, kernel, iterations=1)

    return erd

##################################################################


cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 440)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 1, 1)

        erd = IMG(img)

        contour, area, per, poi, box, rota, points = CONTOUR(erd, img)
        if contour is not None and points is not None:
            cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 5)
            print(f'are:{area}, peri:{per}, poi:{poi}, box:{box}')
            cv2.imshow('img', img)
        else:
            cv2.putText(img, f'NO_CONTOUR', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

