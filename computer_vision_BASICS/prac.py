import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,360)

count = 1

os.makedirs('Images')

while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        cv2.putText(img, f'img_{count}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 3)
        cv2.imshow('img', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        image_name = f'Images/img_{count}.jpg'
        cv2.imwrite(image_name, img)
        count+=1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()