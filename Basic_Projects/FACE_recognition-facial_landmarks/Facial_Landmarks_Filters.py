import cv2
import numpy as np
import dlib

######################################################################################

def create_box(img, point, masked, cropped):
    if masked:
        mask = np.zeros_like(img)
        lips_white = cv2.fillPoly(mask, [point], (255,255,255))
        lips_white_mask = cv2.bitwise_and(mask, lips_white)
        return mask

    if cropped:
        points = cv2.boundingRect(point)
        (x,y,w,h) = points
        box_img = img[y:y+h, x:x+w]
        box_img = cv2.resize(box_img, (0,0), None, 2,2)
        return box_img

######################################################################################

def empty(a):
    pass
cv2.namedWindow('BGR')
cv2.resizeWindow('BGR', 720, 420)
cv2.createTrackbar('Blue', 'BGR', 0, 255, empty)
cv2.createTrackbar('Green', 'BGR', 0, 255, empty)
cv2.createTrackbar('Red', 'BGR', 0, 255, empty)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img = cap.read()

    if img is not None:
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(img_gry)
        for face in faces:
            (x1,y1) = (face.left(), face.top())
            (x2,y2) = (face.right(), face.bottom())
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 5, 0)

            landmarks = predictor(img_gry, face)
            my_points = []
            for n in range(68):
                (x,y) = (landmarks.part(n).x, landmarks.part(n).y)
                # cv2.circle(img, (x,y), 7, (0,0,255), cv2.FILLED)
                # cv2.putText(img, f'{n}', (x,y-3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
                my_points.append([x,y])

            my_points = np.array(my_points[48:68])
            box_img = create_box(img, my_points, masked=True, cropped=False)
            b = cv2.getTrackbarPos('Blue', 'BGR')
            g = cv2.getTrackbarPos('Green', 'BGR')
            r = cv2.getTrackbarPos('Red', 'BGR')
            col = np.zeros_like(img)
            col[:] = (b,g,r)
            red_lips = cv2.bitwise_and(box_img, col)
            red_lips = cv2.GaussianBlur(red_lips, (9,9), 9)

            lips_col_img = cv2.addWeighted(img, 1, red_lips, 0.5, 0)

            img_gry = cv2.cvtColor(img_gry, cv2.COLOR_GRAY2BGR)
            lips_blk_img = cv2.addWeighted(img_gry, 1, red_lips, 0.5, 0)

            hstack = np.hstack([lips_col_img, lips_blk_img])
            cv2.imshow('BGR', hstack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

######################################################################################