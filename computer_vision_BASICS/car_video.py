import cv2

cap = cv2.VideoCapture('car.mp4')

car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

while cap.isOpened():
    ret, img_col = cap.read()
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(img_gry, 1.1, minNeighbors=4, minSize=(30,30))

    for (x,y,a,b) in cars:
        cv2.rectangle(img_col, (x,y), (x+a,y+b), (255,0,255), 3)
        cv2.imshow('output', img_col)

    if cv2.waitKey(1) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
