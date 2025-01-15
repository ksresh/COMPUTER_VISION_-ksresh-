import cv2

def live_video():

    cap = cv2.VideoCapture(0)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

    while cap.isOpened():
        ret, img_col = cap.read()
        img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(img_gry, 1.1, minNeighbors=5, minSize=(30,30))
        for (x,y,a,b) in faces:
            cv2.rectangle(img_col, (x,y), (x+a,y+b), (255,255,0), 5)

            eye_img_col = img_col[y:y+b, x:x+a]
            eye_img_gry = img_gry[y:y+b, x:x+a]
            eyes = eye_classifier.detectMultiScale(eye_img_gry, 1.1, minNeighbors=7, minSize=(15,15))
            for (ex,ey,ea,eb) in eyes:
                cv2.rectangle(eye_img_col, (ex,ey), (ex+ea, ey+eb), (0,255,255), 3)

        cv2.imshow('output', img_col)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


live_video()