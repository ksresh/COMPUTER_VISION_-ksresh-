import cv2
import numpy as np

cap = cv2.VideoCapture('car.mp4')
framewidth, frameheight = (720,460)
# cap.set(3, framewidth)
# cap.set(4, frameheight)

# cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref,frame = cap.read()
    frame = cv2.resize(frame, (framewidth,frameheight))
    # frame = cv2.GaussianBlur(frame, (5,5), 9)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()