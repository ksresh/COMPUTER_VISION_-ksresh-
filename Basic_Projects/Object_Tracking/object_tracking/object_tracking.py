import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ref, img = cap.read()
img = cv2.resize(img, (720,440))

# tracker = cv2.legacy.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()
bo_box = cv2.selectROI('Tracker', img, False)
tracker.init(img, bo_box)

while cap.isOpened():
    timer = cv2.getTickCount()
    ref, img = cap.read()

    if img is not None:
        img = cv2.resize(img, (720,440))
        ref, bo_box = tracker.update(img)
        if bo_box:
            x,y,w,h = int(bo_box[0]), int(bo_box[1]), int(bo_box[2]), int(bo_box[3])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 2)
            cv2.putText(img, f'FACE', (x,y-50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        else:
            cv2.putText(img, f'NONE', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)

        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        cv2.putText(img, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255, 0), 2)
        cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()