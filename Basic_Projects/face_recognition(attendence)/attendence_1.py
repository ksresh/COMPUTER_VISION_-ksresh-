import cv2
import numpy as np
import pandas as pd
import os
import face_recognition
from datetime import datetime


#################################################################################################

path = 'image_attendence'
classes = [nam.split('.')[0] for nam in os.listdir(path) if nam != '.DS_Store']
encodings = []
locations = []
for img in os.listdir(path):
    if img is not None and not img.startswith('.'):  # not None & not starts with '.'
        img_path = os.path.join(path, img)
        img_ary = cv2.imread(img_path)

        encod = face_recognition.face_encodings(img_ary)
        if encod:
            encod = encod[0] #[[4,54,3,5,4]] - 2D to [4,54,3,5,4]-1D
            encodings.append(encod)

        loca = face_recognition.face_locations(img_ary)
        if loca:
            locations.append(loca)

#################################################################################################


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img1 = cap.read()
    if img1 is not None:
        img1 = cv2.resize(img1, (0,0), None, 1,1)

        loca1 = face_recognition.face_locations(img1)
        if loca1:  # to avoid error when there is no face in camera / screen
            (top, left, bottom, right) = loca1[0]   # ex:- (480, 1342, 1035, 788)
            cv2.rectangle(img1, (left,top), (right,bottom), (255,0,0), 5,2)

        my_dis = []
        encod1 = face_recognition.face_encodings(img1)
        if encod1:
            # encod1 = np.array(encod1).flatten()  # FLATTEN from 2D ro 1D
            encod1 = encod1[0]
            for enco in encodings:  # taking encodings of 3 base images to cape with live image
                dis = face_recognition.face_distance([encod1], enco)
                if dis: # to avoid no distance when there is no image in both input and output
                    my_dis.append(dis)

#################################################################################################


        if my_dis:
            mini = min(my_dis)
            if mini < 0.5:
                min_index = np.argmin(my_dis)
                clas_name = classes[min_index]
                date = datetime.now().date()
                time = datetime.now().time().strftime('%H:%M:%S')
                data = []
                if clas_name not in data:   # add all data to 'data' when there is no clas_name
                    cv2.putText(img1, f'{date} : {time}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
                    cv2.putText(img1, f'{clas_name}', (left, top-10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
                    data.append({'name' : clas_name, 'date' : date, 'time' : time})
        else:
            cv2.putText(img1, f'NONE', (50,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

    cv2.imshow('img', img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if data:
            df = pd.DataFrame(data)
            df.to_csv('attendence.csv', mode='a')
            break

cap.release()
cv2.destroyAllWindows()