import cv2
import numpy as np
import zxing

cap = cv2.VideoCapture(0)
cap.set(4,900)
cap.set(3,700)

reader = zxing.BarCodeReader()

while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        cv2.imwrite('image.png', img) # creating image from img
        text = reader.decode('image.png')  # reading saved image

        if text:
            point = text.points
            (x,y,w,h) = (int(point[3][0]), int(point[3][1]), int(point[1][0]), int(point[1][1]))
            print(text)
            cv2.rectangle(img, (x,y), (w,h), (255,0,0), 2,2)
            cv2.putText(img, f'{text.raw}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2 )
        else:
            print('None')

        cv2.imshow('img', img)
        # cv2.putText(img, f'{text}', )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()