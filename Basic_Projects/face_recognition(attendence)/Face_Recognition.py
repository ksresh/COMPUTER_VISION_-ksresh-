'''import cv2
import numpy as np
import face_recognition

sathya_img = face_recognition.load_image_file('image_basic/sathya_nadella_1.png')
sathya_img = cv2.cvtColor(sathya_img, cv2.COLOR_RGB2BGR)

sathya_test = face_recognition.load_image_file('image_basic/sathya_nadella_2.png')
sathya_test = cv2.cvtColor(sathya_test, cv2.COLOR_RGB2BGR)

img_fac_loc = face_recognition.face_locations(sathya_img)   # location of face ([(116, 683, 270, 528)]) = (top, right, bottom, left)
(img_top, img_right, img_bottom, img_left) = img_fac_loc[0]   # = (116, 683, 270, 528)
img_fac_end = face_recognition.face_encodings(sathya_img)
img_fac_end = img_fac_end[0]
cv2.rectangle(sathya_img, (img_left, img_top), (img_right, img_bottom), (0,255,0), 2, 2)


test_fac_loc = face_recognition.face_locations(sathya_test)  # 1st finding
(test_top, test_right, test_bottom, test_left) = test_fac_loc[0]
test_fac_end = face_recognition.face_encodings(sathya_test)
test_fac_end = test_fac_end[0]
cv2.rectangle(sathya_test, (test_left, test_top), (test_right, test_right), (0,0,255), 2, 2)


result1 = face_recognition.compare_faces([img_fac_end], test_fac_end)  # true / false
distance1 = face_recognition.face_distance([img_fac_end], test_fac_end)  # less_distance = more accuracy
print(result1, distance1)

cv2.putText(sathya_test, f'{result1[0]}, {round(distance1[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
# result1=[True] & result1[0]=True and same for distance1[0]

cv2.imshow('sathya_img', sathya_img)
cv2.imshow('sathya_test', sathya_test)
cv2.waitKey(0)'''


#################################################################################################

import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime
import pandas as pd


def loc_end():
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
            fac_loc = face_recognition.face_locations(image)
            fac_locations.append(fac_loc)
            fac_end = face_recognition.face_encodings(image)
            fac_encodings.append(fac_end)
    return classes, images, fac_locations, fac_encodings

#################################################################################################

def video():
    classes, images, fac_locations, fac_encodings = loc_end()

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4, 440)


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
                my_distances = []
                for fac_end in fac_encodings:
                    fac_end = fac_end[0]
                    dis = face_recognition.face_distance([end], fac_end)
                    my_distances.append(dis)

                if min(my_distances) < 0.5:
                    low = np.argmin(my_distances)
                    class_name = classes[low]
                    cv2.putText(img, f'{class_name}', (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    date = datetime.now().date()
                    time = datetime.now().time().strftime('%H:%M:%S')

                else:
                    cv2.putText(img, f'NONE', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return class_name, date, time

#################################################################################################
my_df = []
class_name, date, time = video()
df = class_name, date, time
my_df.append(df)
df =pd.DataFrame(columns = ['name' , 'date', 'time'], data=my_df)
df.to_csv('attendance.csv', mode='a', header=not os.path.join('attendance.csv'), index=False)
