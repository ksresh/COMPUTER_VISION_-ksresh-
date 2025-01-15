import cv2

cap = cv2.VideoCapture('people_walking.mp4')

body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_fullbody.xml')

while cap.isOpened():
    ret, img_col = cap.read()
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

    body = body_classifier.detectMultiScale(img_gry, 1.6, minNeighbors=2)
    for (x,y,a,b) in body:
        cv2.rectangle(img_col, (x,y), (x+a,y+b), (255,0,255), 4)
        cv2.imshow('output', img_col)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cap.release()
cv2.destroyAllWindow()

