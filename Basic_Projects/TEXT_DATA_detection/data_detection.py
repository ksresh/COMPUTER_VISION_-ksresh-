import cv2
import numpy as np
import pytesseract

# for mac 'brew install tesseract'  or download tesseract file save in your system
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

img = cv2.imread('mine.png')
# img = cv2.imread('mine.png')   '''store some pic which have some data in this main folder and read this'''

string = pytesseract.image_to_string(img)
box = pytesseract.image_to_boxes(img)

config = '--oem 3 --psm 6 outputbase tessdata/eng'
data = pytesseract.image_to_data(img, config=config)

for datas in data.splitlines():
    datas = datas.split()

    if len(datas) == 12 and datas[11].isalpha() and datas[11] != 'text':
        (x,y,w,h,word) = int(datas[6]), int(datas[7]), int(datas[8]), int(datas[9]), datas[11],
        cv2.rectangle(img, (x,y), (w+x,h+y), (255,0,255), 3, 1)
        cv2.putText(img, f'{word}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 2)
        print(datas)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

