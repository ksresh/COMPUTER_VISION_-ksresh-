import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)  # (0 - 1) range
# higher 'detectionCon' higher strict correct exact detection of hand = 0.9
# lowe 'detectionCon' higher detection all hands including non-hands some-time wrong detection = 0.5

########################################################################################

def HANDS(img):
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]  # dictionary{lmList - bbox - center - type}
        lmList1 = hand1['lmList']
        p1, p2  = lmList1[4][0:2], lmList1[8][0:2] # [245, 67, 87] = we need only first 2 numbers
        dis1, info1, img = detector.findDistance(p1, p2, img)
        cv2.putText(img, f'{int(dis1)}', (info1[4:]), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 5)
        for list1 in lmList1:
            (x1,y1) = (list1[0], list1[1])
            cv2.circle(img, (x1,y1), 15, (0,0,255), cv2.FILLED)

        finger1= detector.fingersUp(hand1)
        print(finger1)  # [1, 1, 0, 0, 0] = [thump, than , middle, than, little] fingers

    if len(hands)==2:
        hand2 = hands[1]  # dictionary{lmList - bbox - center - type}
        lmList2 = hand2['lmList']
        p3, p4  = lmList2[4][0:2], lmList2[8][0:2]  # [245, 67, 87] = we need only firt 2 numbers
        dis2, info2, img = detector.findDistance(p2, p4, img)
        cv2.putText(img, f'{int(dis2)}', (info2[4:]), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        for list2 in lmList2:
            (x2,y2) = (list2[0], list2[1])
            cv2.circle(img, (x2,y2), 15, (0,0,255), cv2.FILLED)
    return img

########################################################################################

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (0,0), None, 2,2)

        img = HANDS(img)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

########################################################################################