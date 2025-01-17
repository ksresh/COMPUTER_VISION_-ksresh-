import cv2
import numpy as np

kernel = np.ones((5,5), dtype=np.uint8)

# img = cv2.imread('paa1.jpg')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ref,img = cap.read()

    if len(img) == 0:
        print('no image')
    else:
        sze_img = cv2.resize(img, (1000,1000))
        gry_img = cv2.cvtColor(sze_img, cv2.COLOR_BGR2GRAY)
        gsb_img = cv2.GaussianBlur(gry_img, (5,5),3)
        can_img = cv2.Canny(gry_img, 100,100)
        dil_img = cv2.dilate(can_img, kernel, 1)
        erd_img = cv2.erode(dil_img, kernel, 5)




        #cv2.imshow('sze_img', sze_img)
        #cv2.imshow('gry_img', gry_img)
        #cv2.imshow('gsb_img', gsb_img)
        #cv2.imshow('can_img', can_img)
        #cv2.imshow('dil_img', dil_img)
        #cv2.imshow('erd_img', erd_img)

        gry_img = cv2.cvtColor(gry_img, cv2.COLOR_GRAY2BGR)
        gsb_img = cv2.cvtColor(gsb_img, cv2.COLOR_GRAY2BGR)
        can_img = cv2.cvtColor(can_img, cv2.COLOR_GRAY2BGR)
        dil_img = cv2.cvtColor(dil_img, cv2.COLOR_GRAY2BGR)
        erd_img = cv2.cvtColor(erd_img, cv2.COLOR_GRAY2BGR)


        h_img1 = np.hstack((sze_img, gry_img, gsb_img))
        h_img2 = np.hstack((can_img, dil_img, erd_img))
        v_img = np.vstack((h_img1, h_img2))
        print(v_img)

    cv2.imshow('v_img', v_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()