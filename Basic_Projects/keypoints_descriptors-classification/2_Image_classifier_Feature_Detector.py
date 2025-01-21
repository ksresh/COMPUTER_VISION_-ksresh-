import cv2
import numpy as np
import os

########################################################################################

orb = cv2.ORB_create()
bf = cv2.BFMatcher()

########################################################################################

def IMG2():
    folder = 'query'
    Class = [clas.split('.')[0] for clas in os.listdir(folder) if clas != '.DS_Store']
    IMG2 = []
    KP2 = []
    DES2 = []
    IMG2_KP2 = []
    for img in os.listdir(folder):
        img2 = os.path.join(folder, img)
        img2 = cv2.imread(img2)
        if img2 is not None:
            kp2, des2 = orb.detectAndCompute(img2, None)
            img2_kp2 = cv2.drawKeypoints(img2, kp2, None)
            IMG2.append(img2)
            KP2.append(kp2)
            DES2.append(des2)
            IMG2_KP2.append(img2_kp2)

    DES2 = np.array(DES2)

    return Class, KP2, DES2

########################################################################################

def COMMON_POINT(Class):
    COUNT = []
    for des in DES2:
        MATCH = []
        matches = bf.knnMatch(des, des1, k=2)
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                MATCH.append(m)
        COUNT.append(len(MATCH))

    index = np.argmax(COUNT)
    prediction = Class[index]
    if COUNT[index] >= 10:
        cv2.putText(img1, f'{prediction}', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
    else:
        cv2.putText(img1, f'NONE', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

    return prediction

########################################################################################

cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4, 600)

while cap.isOpened():
    ref, img1 = cap.read()
    if img1 is not None:
        kp1, des1 = orb.detectAndCompute(img1, None)
        img1_kp1 = cv2.drawKeypoints(img1, kp1, None)

        Class, KP2, DES2 = IMG2()

        prediction = COMMON_POINT(Class)

    cv2.imshow('img1', img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






