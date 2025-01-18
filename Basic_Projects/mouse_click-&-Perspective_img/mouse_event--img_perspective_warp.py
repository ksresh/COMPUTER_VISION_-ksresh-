import cv2
import numpy as np

img =  cv2.imread('mamatha.png')
##################################################################

def POINTS():
    points = []
    def EVENT(event, x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x,y])

    cv2.imshow('img', img)
    cv2.setMouseCallback('img', EVENT)
    cv2.waitKey(0)
    return points

##################################################################

def POINTS_pts2(pts1, img):
    if len(pts1) == 4:
        for pt in pts1:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (255,0,0), cv2.FILLED)
            # cv2.imshow('img', img)

##################################################################

points = POINTS()
pts1 = np.array(points, dtype=np.float32)
pts2 = np.array([[0,0], [500,0], [500,500], [0,500]], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(pts1, pts2)
warp_img = cv2.warpPerspective(img, matrix, (500,500))

POINTS_pts2(pts1, img)
cv2.imshow('img', img)
cv2.imshow('warp_img', warp_img)

cv2.waitKey(0)



##################################################################
