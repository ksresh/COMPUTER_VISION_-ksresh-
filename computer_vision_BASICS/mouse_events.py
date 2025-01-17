import cv2
import numpy as np

img = cv2.imread('books.png')

def mouse_event(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

p1 = np.float32([[299,46],[138,259],[640,422],[774,189]])
p2 = np.float32([[400,0],[0,0],[0,600],[400,600]])

matrix = cv2.getPerspectiveTransform(p1, p2)
output = cv2.warpPerspective(img, matrix, (400,600))

for x in range(0,4):
    cv2.circle(img, (int(p1[x][0]), int(p1[x][1])), 5, (255,0,255), cv2.FILLED)


cv2.imshow('img', img)
cv2.imshow('output', output)
cv2.setMouseCallback('img', mouse_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

