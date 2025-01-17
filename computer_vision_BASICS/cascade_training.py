import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,300)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def empty(a):
    pass
cv2.namedWindow('detect')
cv2.resizeWindow('detect', 1200,600)
cv2.createTrackbar('scale', 'detect', 11,100,empty)
cv2.createTrackbar('neighbours', 'detect', 1,10,empty)
cv2.createTrackbar('area', 'detect', 1000,30000, empty)

while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        display = np.ones((600,1200,3), dtype=np.uint8)

        img = cv2.resize(img, (600,300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        scale1 = cv2.getTrackbarPos('scale', 'detect')/10
        neighbours1 = cv2.getTrackbarPos('neighbours', 'detect')

        face = face_cascade.detectMultiScale(gray, scale1, neighbours1)
        for (x,y,w,h) in face:
            area = w*h
            area1 = cv2.getTrackbarPos('area', 'detect')
            if area > area1:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255),3)
                cv2.putText(img, 'FACE', (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 3)

        display[100:400, :600] = img
        display[100:400, 600:1200] = gray_3d

        cv2.imshow('detect', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()