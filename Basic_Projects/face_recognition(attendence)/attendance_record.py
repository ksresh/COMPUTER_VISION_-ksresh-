import cv2
import numpy as np
import pandas as pd
import os
#from pyforest import *
import face_recognition
from datetime import datetime



######################################################################################
def base():
    path = 'image_attendence'
    classes = [name.split('.')[0] for name in os.listdir(path) if name != '.DS_Store']
    images = []
    fac_locations = []
    fac_encodings = []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)

            loc = face_recognition.face_locations(image)
            if loc:
                fac_locations.append(loc)

            end = face_recognition.face_encodings(image)
            if end:
                fac_encodings.append(end)

    return classes, images, fac_locations, fac_encodings

######################################################################################

def video():
    classes, images, fac_locations, fac_encodings = base()

    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(4, 420)

    while cap.isOpened():
        ref, img = cap.read()
        if img is not None:

            loc = face_recognition.face_locations(img)
            if loc:
                (top, right, bottom, left) = loc[0]
                cv2.rectangle(img, (left,top), (right,bottom), (0,255,0), 2, 2)

            end = face_recognition.face_encodings(img)
            if end:
                end = end[0]
                my_dis = []
                for fac_end in fac_encodings:
                    fac_end = fac_end[0]
                    dis = face_recognition.face_distance([end], fac_end)
                    my_dis.append(dis)

                if min(my_dis) < 0.6:
                    mini = np.argmin(my_dis)
                    class_name = classes[mini]
                    cv2.putText(img, f'{class_name}', (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    date = datetime.now().date()
                    time = datetime.now().strftime('%H:%M:%S')   # 'strftime' = string format time
                    cv2.putText(img, f'{time}', (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

                else:
                    cv2.putText(img, f'NONE', (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return class_name, date, time

######################################################################################

class_name, date, time  = video()

if class_name:
    df = pd.DataFrame({'name' : [class_name], 'date' : [date], 'time' : [time]})
    df.to_csv('attendence1.csv', mode='a', header=not os.path.join('attendence.csv'), index=True)

    print(df)

######################################################################################

