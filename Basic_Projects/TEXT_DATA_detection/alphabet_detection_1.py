import cv2
import pytesseract
import numpy as np

# for mac 'brew install tesseract'  or download tesseract file save in your system
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

########################################################################################

def BOXES(img):

    my_data = []

    boxes = pytesseract.image_to_boxes(img) # read the image to alphabet

    for box in boxes.splitlines():  # splitting lines
        box1 = box.split()  # to convert int0 array types text list

        # (x,y)=bottom left points   &  (w,h)=top right corners
        (text, x,y,w,h) = (box1[0]), int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
        cv2.rectangle(img, (x,600-y), (w, 600-h), (0,0,255), 1,1)   # use (width,height)=(1000,600)
        cv2.putText(img, f'{text}', (x,600-y-30), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 1)

        my_data.append((text,x,y,w,h))

    return my_data

########################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 600)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:

        # img = cv2.imread('mine.png')   '''store some pic which have some data in this main folder and read this'''

        my_data =  BOXES(img)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()