'''import cv2
import numpy as np

img1 = cv2.imread('query/car1.png', 0)
img2 = cv2.imread('train/nfs.png', 0)

orb =cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img1_kp1 = cv2.drawKeypoints(img1, kp1, None)
img2_kp2 = cv2.drawKeypoints(img2, kp2, None)

good = []
bfm = cv2.BFMatcher()
matchers = bfm.knnMatch(des1, des2, k=2)
for m,n in matchers:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow('img3', img3)
cv2.waitKey(0)'''

################################################################

import cv2
import numpy as np
import os

def query_base():
    path = 'query'
    images = []
    kp_qs = []
    des_qs = []
    classes = [name.split('.')[0] for name in os.listdir(path) if name != '.DS_Store']
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is not None:
            image_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)

            orb = cv2.ORB_create()
            kp_q, des_q = orb.detectAndCompute(image_gry, None)
            kp_qs.append(kp_q)
            des_qs.append(des_q)
    return images, classes, kp_qs, des_qs

################################################################

images, classes, kp_qs, des_qs = query_base()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 440)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img_org = img.copy()
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img_gry, None)

        my_matches = []
        for des_q in des_qs:
            good = []
            bfm = cv2.BFMatcher()
            matchers = bfm.knnMatch(des, des_q, k=2)
            for m,n in matchers:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            my_matches.append(len(good))
        print(my_matches)

        if my_matches:
            max = np.argmax(my_matches)
            max_match = my_matches[max]
            class_name = classes[max]

            if max_match > 10:
                cv2.putText(img, f'{class_name}', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
            else:
                cv2.putText(img, f'NONE', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

    cv2.imshow('img3', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

