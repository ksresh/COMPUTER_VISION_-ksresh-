import cv2
import numpy as np

import pytesseract
# for mac 'brew install tesseract'  or download tesseract file save in your system
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

img = cv2.imread('mine.png')
# img = cv2.imread('mine.png')   '''store some pic which have some data in this main folder and read this'''

img = cv2.resize(img, (1200,800))

string = pytesseract.image_to_string(img)
box = pytesseract.image_to_boxes(img)

height, width = 800,1200

for boxes in box.splitlines():
    boxes = boxes.split(' ')
    print(boxes)

    text,x,y,w,h= str(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
    cv2.rectangle(img, (x,height-y), (w,height-h), (255,0,255),2,2)
    cv2.putText(img, f'{text}', (x,height-y+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),1)
    print(x,y,w,h)
    print(boxes)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()