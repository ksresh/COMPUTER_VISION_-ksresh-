# FACE_EYES_detection(photo)
def face_eyes(image):
    import cv2

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

    while True:
        img_col = cv2.imread(image)
        img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(img_gry, 1.5, minNeighbors=5, minSize=(30,30))

        for (x,y,a,b) in faces:
            cv2.rectangle(img_col, (x,y), (x+a,y+b), (255,255,0), 3)
            cv2.imshow('output', img_col)

            eye_img_col = img_col[y:y+b, x:x+a]
            eye_img_gry = img_gry[y:y+b, x:x+a]
            eyes = eye_classifier.detectMultiScale(eye_img_gry, 1.01, minNeighbors=6)

            for (ex,ey,ea,eb) in eyes:
                cv2.rectangle(eye_img_col, (ex,ey), (ex+ea, ey+eb), (0,255,255), 2)
        cv2.imshow('output', img_col)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


face_eyes('paa1.jpg')


#-------------------------------------------------------------------------------------------------#

# CAR_detection(video)


import cv2

cap = cv2.VideoCapture('car.mp4')
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

while cap.isOpened():
    ref, vid_col = cap.read()
    vid_gry = cv2.cvtColor(vid_col, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(vid_gry, 1.2, minNeighbors=4, minSize=(30,30))

    for (x,y,a,b) in cars:
        cv2.rectangle(vid_col, (x,y), (x+a, y+b), (255,0,255), 3)

    cv2.imshow('output', vid_col)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------#

import cv2

cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')


while cap.isOpened():
    ref, vid_col = cap.read()
    vid_gry = cv2.cvtColor(vid_col, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(vid_gry, 1.3, minNeighbors=5, minSize=(30,30))

    for (x,y,a,b) in faces:
        cv2.rectangle(vid_col, (x,y), (x+a, y+b), (255,255,0), 3)
        cv2.imshow('face_eye', vid_col)

        eye_vid_col = vid_col[y:y+b, x:x+a]
        eye_vid_gry = vid_gry[y:y+b, x:x+a]
        eyes = eye_classifier.detectMultiScale(eye_vid_gry, 1.2, minNeighbors=6, minSize=(20,20))

        for (ex,ey,ea,eb) in eyes:
            cv2.rectangle(eye_vid_col, (ex,ey), (ex+ea, ey+eb), (0,255,255), 2)

    cv2.imshow('face_eye', vid_col)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------#

import cv2


