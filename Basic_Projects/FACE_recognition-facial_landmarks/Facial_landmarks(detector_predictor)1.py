import cv2
import numpy as np
import dlib

def empty(a):
    pass
cv2.namedWindow('hstack')
cv2.resizeWindow('hstack', 1000, 600)
cv2.createTrackbar('h_stack', 'hstack', 0, 255, empty)

######################################################################################

def IMG(img):  # convert img to gray and its copy
    img = cv2.resize(img, (0, 0), None, 1.5, 1.5)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gry_copy = img_gry.copy()
    img_gry_copy = cv2.cvtColor(img_gry_copy, cv2.COLOR_GRAY2BGR)
    return img, img_gry, img_gry_copy

######################################################################################

def FACE(img, img_gry):  # face & marks detection
    detector = dlib.get_frontal_face_detector()
    faces = detector(img)
    for face in faces:  # [(753, 496) (1523, 1266)]
        (x, y, w, h) = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(img, (x,y), (w,h), (255,0,0), 5,1)

        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_marks = predictor(img_gry, face)
        my_points = []
        for n in range(68):
            (X, Y) = (face_marks.part(n).x, face_marks.part(n).y)
            my_points.append([X, Y])
            cv2.circle(img, (X,Y), 5, (0,0,255), cv2.FILLED)
            cv2.putText(img, f'{n}', (X-3,Y-3), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)
        return my_points
    return None

######################################################################################

def LIPS(img_gry, my_points):  # lips filling
    if my_points is not None and len(my_points) >= 68:
        fill_points = np.array(my_points[48:68])
        mask = np.zeros_like(img_gry_copy) # mask creation same as img_gry_copy
        red = cv2.getTrackbarPos('h_stack', 'hstack')
        lip_red = cv2.fillPoly(mask, [fill_points], (0, 0, red))
        lip_red_mask = cv2.bitwise_and(mask, lip_red)
        LIP_COL = cv2.addWeighted(lip_red_mask, 0.5, img_gry_copy, 1, 0)

        hstack = np.hstack([img, LIP_COL])
        cv2.imshow('hstack', hstack)

######################################################################################

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:

        img, img_gry, img_gry_copy = IMG(img)

        my_points = FACE(img, img_gry)

        LIPS(img_gry, my_points)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

######################################################################################