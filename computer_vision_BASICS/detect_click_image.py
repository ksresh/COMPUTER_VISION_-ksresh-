import cv2
import numpy as np

img = cv2.imread('books.png')

circles = np.ones((4,2), dtype=np.float32)
count = 0

while True:
    def mouse_event(event,x,y,flags,params):
        global count
        if event == cv2.EVENT_LBUTTONDOWN:
            circles[count] = [x,y]
            count = count +1

            if count == 4:
                p1 = circles.copy()
                p2 = np.float32([[400,0],[0,0],[0,600],[400,600]])
                matrix = cv2.getPerspectiveTransform(p1,p2)
                output = cv2.warpPerspective(img, matrix, (400,600))

                for x in range(0,4):
                    cv2.circle(img, (int(p1[x][0]), int(p1[x][1])), 5, (255,0,255), cv2.FILLED)
                cv2.imshow('output', output)

    cv2.imshow('img', img)
    cv2.setMouseCallback('img', mouse_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()