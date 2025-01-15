def live_video():
    import cv2

    cap = cv2.VideoCapture(0)  # capturing video

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_classifier  = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

    while cap.isOpened():
        ref, img_col = cap.read()  # extracting frames from video
        img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY) # converting frame to gray image
        faces = face_classifier.detectMultiScale(img_gry, 1.1, minNeighbors=5, minSize=(30,30))  # finding faces

        for (x,y,a,b) in faces:
            cv2.rectangle(img_col, (x,y), (x+a,y+b), (255,255,0), 5)  # drawing rectangle around face

            eye_img_col = img_col[y:y+b, x:x+a]  # defing new colour image with new measurement for eyes
            eye_img_gry = img_gry[y:y+b, x:x+a] # defing new gray image with new measurement for eyes
            eyes = eye_classifier.detectMultiScale(eye_img_gry, 1.04, minNeighbors=8, minSize=(20,20))  # finding eyes

            for (ex,ey,ea,eb) in eyes:
                cv2.rectangle(eye_img_col, (ex,ey), (ex+ea,ey+eb), (0,255,255), 3)  # drawing rectangle around eyes

        cv2.imshow('face & eye', img_col) # ploting final images
        if cv2.waitKey(1) & 0xFF == ord('q'):   # break or close cemara tab when i press ENTER or Q letter key
            break

    cap.release()  # release all the objects that we have from video
    cv2.destroyAllWindows()  # close or destroy all pop up tabs in the  end

live_video()