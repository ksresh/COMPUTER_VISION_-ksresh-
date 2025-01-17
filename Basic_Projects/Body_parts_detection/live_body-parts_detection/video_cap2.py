import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

####################################################################

DIRECTORY = 'Body_parts_detection'
CATEGORIES = [cat for cat in os.listdir(DIRECTORY) if cat!='.DS_Store']

####################################################################

def IMG(img):
    img = cv2.resize(img, (256, 256))
    img = img / 256
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img, verbose=0)
    index = np.argmax(predict)
    class_name = CATEGORIES[index]
    percentage = predict[0][index]*100
    return class_name, percentage

####################################################################

def CONTOUR(img):
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gry, (9,9), 1)
    cny = cv2.Canny(blr, 100, 100)

    kernel = np.ones((5,5))
    dlt = cv2.dilate(cny, kernel, iterations=2)
    erd = cv2.erode(dlt, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(erd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            per = cv2.arcLength(contour, closed=True)
            poi = cv2.approxPolyDP(contour, 0.2*per, closed=True)
            box = cv2.boundingRect(contour)
            return box, contour
    return None


####################################################################

cap = cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,440)

model = load_model('vgg16_model1.h5')

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img_copy = img.copy()

        class_name, percentage = IMG(img)
        if percentage > 50:
            result = CONTOUR(img)
            if result :
                box, contour = result
                (x,y,w,h) = box
                cv2.drawContours(img_copy, [contour], -1, (255,255,0), 2, cv2.FILLED)
                cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,255), 2, 1)
                cv2.putText(img_copy, f'{percentage:.2f} : {class_name}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        else:
            cv2.putText(img_copy, f'NONE', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        cv2.imshow('img_copy', img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


####################################################################