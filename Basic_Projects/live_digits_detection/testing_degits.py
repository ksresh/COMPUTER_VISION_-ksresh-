import cv2
import numpy as np
import tensorflow
import os

cap = cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,420)

from tensorflow.keras.models import load_model
model = load_model('model_vgg16.h5')

DIRECTORIES = r'digits'
CATEGORIES = os.listdir(DIRECTORIES)
CATEGORIES = [cat for cat in CATEGORIES if cat != '.DS_Store']

while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        img_copy = img.copy()

        img = cv2.resize(img, (256,256))
        img = img/256
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=1)
        clas = np.argmax(pred)
        name = CATEGORIES[clas]

        per = np.round(pred[0][clas]*100, 2)

        cv2.putText(img_copy, f'{name}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,255), 1)
        cv2.putText(img_copy, f'{per}%', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,255), 1)


        cv2.imshow('img_copy', img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()