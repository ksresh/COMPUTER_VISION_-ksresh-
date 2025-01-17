import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
cap.set(3,620)
cap.set(4,320)

os.makedirs('Images')

count = 1
while cap.isOpened():
    ref, img =cap.read()

    if img is not None:
        cv2.putText(img, f'img_{count}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image = f'Images/img_{count}.jpg'
        cv2.imwrite(image, img)
        print(f'img_{count}')
        count += 1

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
