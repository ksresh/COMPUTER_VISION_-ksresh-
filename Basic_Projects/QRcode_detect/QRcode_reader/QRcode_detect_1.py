import cv2
import numpy as np
import zxing

reader = zxing.BarCodeReader()

cap = cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,440)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 1,1)

        image = cv2.imwrite('image.png', img)  # creating image from img
        text = reader.decode('image.png')  # reading saved image
        if text:
            p3 = (int(text.points[3][0]), int(text.points[3][1]))
            p1 = (int(text.points[1][0]), int(text.points[1][1]))
            cv2.rectangle(img, p3, p1, (255,0,0), 2, 2)
            cv2.putText(img, f'{text.raw}', (int(text.points[3][0]), int(text.points[3][1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)


        else:
            cv2.putText(img, f'NONE', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 1)


    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()