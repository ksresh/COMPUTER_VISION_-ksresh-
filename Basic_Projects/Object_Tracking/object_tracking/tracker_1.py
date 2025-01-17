import cv2
import numpy as np

#############################################################################
cap = cv2.VideoCapture(0)
cap.set(3,900)
cap.set(4, 500)

ref,img1 = cap.read()

#############################################################################

tracker = cv2.TrackerCSRT_create()
bo_box = cv2.selectROI('tracker', img1, False)
tracker.init(img1, bo_box)

#############################################################################

while cap.isOpened():
    ref, img = cap.read()
    count = cv2.getTickCount()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 1,1)

        __ , bo_box = tracker.update(img)
        (x,y) = (int(bo_box[0]), int(bo_box[1]))
        (w,h) = (int(bo_box[2]), int(bo_box[3]))
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2,1)
        cv2.putText(img, f'FACE', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0), 2)

        fps = cv2.getTickFrequency()//(cv2.getTickCount()-count)
        cv2.putText(img, f'{fps}', (x+w+20, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()