import cv2
import numpy as np
import pytesseract

# for mac 'brew install tesseract'  or download tesseract file save in your system
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

'''strings = pytesseract.image_to_data(img)'''

########################################################################################

def DATA(img):
    config = '--oem 3 --psm 6 outputbase tessdata/eng'
    data = pytesseract.image_to_data(img, config=config)
    my_data = []
    for dat in data.splitlines():  # spliting data by lines
        dat1 = dat.split()  # spliting to convert into array format list

        if len(dat1)==12 and not dat1[6].isalpha():  # for all data & removing text row from 6 column
        # if len(dat1)==12 and not dat1[6].isalpha() and dat1[-1].isalpha():   # for only alphabets
        # if len(dat1) == 12 and not dat1[6].isalpha() and not dat1[-1].isalpha():  # for only other than alphabets
            (x,y,w,h,text) = (int(dat1[6]), int(dat1[7]), int(dat1[8]), int(dat1[9]), dat1[11])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 2)
            cv2.putText(img, f'{text}', (x,y-3), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255,0,0), 2)
            my_data.append((x,y,w,h,text))

    print(my_data)
    return my_data

########################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 600)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:

        # img = cv2.imread('mine.png')   '''store some pic which have some data in this main folder and read this'''

        my_data = DATA(img)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

