import cv2
import numpy as np
import os



''' Coverting best features on images into computer language called descriptors
An than we match 2 images'''

'''we use ORB algorithm is fast, accurate, free to find descriptors of image'''

'''img1 = cv2.imread('query/car1.png', 0)
img2 = cv2.imread('train/car.png', 0)

orb = cv2.ORB_create(nfeatures=1000) # fast, free algoritms, nfeatures can increase or decrease
# nfeatures = number of features you want

kp1, des1 = orb.detectAndCompute(img1, None)  # None = no mask
kp2, des2 = orb.detectAndCompute(img2, None)  # des1/des2 = (500, 32), 500 features, 32 values

img1_kp1 = cv2.drawKeypoints(img1, kp1, None) # output_image = None
img2_kp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()  # bf = brute fruit matcher
matchers = bf.knnMatch(des1, des2, k=2)  # 2 matches = m,n

good = []
for m,n in matchers:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2) # output

cv2.imshow('img3', img3)

cv2.imshow('img1_kp1', img1_kp1)
cv2.imshow('img2_kp2', img2_kp2)

cv2.imshow('car1', img1)
cv2.imshow('jurrasic1', img2)

cv2.waitKey(0)'''

########################################################################################

orb = cv2.ORB_create()
bf = cv2.BFMatcher()


cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 600)

while cap.isOpened():
    ref, img2 = cap.read()
    if img2 is not None:

        kp2, des2 = orb.detectAndCompute(img2, None)
        img2_kp2 = cv2.drawKeypoints(img2, kp2, None)

########################################################################################

        folder = 'train'
        Class = [cat.split('.')[0] for cat  in os.listdir(folder) if cat != '.DS_Store']
        # removing '.DS_Store' temp file from folder

        IMG1 = []  # all images from folder
        KP1 = []  # all keypoints of all images from folder
        DES1 = []  # all descriptors of all images from folder
        IMG1_KP1 = []  # all keypoint diagrams of all images from folder
        for image1 in os.listdir(folder):
            img1 = os.path.join(folder, image1)
            img1 = cv2.imread(img1, 0)

            kp1, des1 = orb.detectAndCompute(img1, None)
            img1_kp1 = cv2.drawKeypoints(img1, kp1, None)

            IMG1.append(image1)
            KP1.append(kp1)
            DES1.append(des1)
            IMG1_KP1.append(img1_kp1)

########################################################################################

        compare = [] # length of good matches of all images with live image
        for des in DES1:
            good_matches = []
            matches = bf.knnMatch(des2, des, k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
            compare.append(len(good_matches))

        class_no = np.argmax(compare)  # index of highest best length of matches
        class_name = Class[class_no]
        if compare[class_no] >=10:
            cv2.putText(img2, f'{class_name}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        else:
            cv2.putText(img2, f'NONE', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)

    cv2.imshow('img2', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



