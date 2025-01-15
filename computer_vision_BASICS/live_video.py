import cv2

cap = cv2.VideoCapture(0)  # capturing video, DSHOW = when you get error

while cap.isOpened():
    ref, frame = cap.read()  # ref=booleans, frame=pictures

    if ref == True:  # only ref/boolean is true
        img_col = cv2.resize(frame, (500, 400))  # resizing the video's frame
        img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)  # converting to gray

        cv2.imshow('COLOR', img_col)
        cv2.imshow('GRAY', img_gry)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press to stop
            break

cap.release()  # release the video
cv2.destroyAllWindows()
