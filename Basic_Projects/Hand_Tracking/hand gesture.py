import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)

cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4, 700)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:

        hands, img = detector.findHands(img)

        if hands and len(hands)==2: # dictionary(lmList, bbox, center, type)
            hand1 = hands[0]
            upfingers1 = detector.fingersUp(hand1)
            lmList1 = hand1['lmList']
            bbox1 = hand1['bbox']
            center1 = hand1['center']
            type1 = hand1['type']
            # cv2.putText(img, f'{type1}', (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)

            # lmList1 are in 3d = '[461, 506, -167]' so we need only 2d
            p1 = (x1,y1) = (lmList1[8][0], lmList1[8][1])
            p2 = (x2,y2) = (lmList1[12][0], lmList1[12][1])

            if len(hands)==2:
                hand2 = hands[1]
                lmList2 = hand2['lmList']
                bbox2 = hand2['bbox']
                center2 = hand2['center']
                type2 = hand2['type']

                p3 = (x3, y3) = lmList2[8][0], lmList2[8][1]
                p4 = (x4, y4) = lmList2[12][0], lmList2[12][1]
                len2, info2, img = detector.findDistance(p1, p3, img)

                cv2.putText(img, f'{int(len2)}', (int((x1+x3)/2)-100, int((y1+y3)/3)+150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)



        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




