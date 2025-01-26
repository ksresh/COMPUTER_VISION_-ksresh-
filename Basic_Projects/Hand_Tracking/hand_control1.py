import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)

##############################################################################

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 2,2)

        hands, img = detector.findHands(img, flipType=True)
        if hands: # dictionary (lmList - bbox - center - type)
            hand1 = hands[0]
            lmList1 = hand1['lmList']
            p1 = lmList1[8][0:2]   # [244, 56, -60]   - we want only first 2 numbers
            p3 = lmList1[4][0:2]
            for list1 in lmList1:
                (x1,y1) = (list1[0], list1[1]) # defining (x,y) to draw circle in every point in hand
                cv2.circle(img, (x1,y1), 20, (255,0,255), cv2.FILLED) # drawing circles

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2['lmList']
            p2 = lmList2[8][0:2]
            for list2 in lmList2:
                (x2,y2) = (list2[0], list2[1])
                cv2.circle(img, (x2,y2), 20, (255,0,255), cv2.FILLED)

                dis, info, img = detector.findDistance(p1, p3, img)
                if info:
                    mid  = (X,Y) = info[4:]
                    cv2.putText(img, f'{int(dis)}', (X-50, Y-50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)



        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()