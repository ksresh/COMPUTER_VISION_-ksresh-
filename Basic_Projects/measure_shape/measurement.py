import cv2
import numpy as np
import utils

##############################################################################

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 420)

def video(cam):
    if cam:
        while cap.isOpened():
            ref, img = cap.read()
            if img is not None:
                img = cv2.resize(img, (0,0), None, 1.5, 1.5)
                df_sort = utils.util(img)
            if cv2.waitKey(1) & 0XFF == ord('q') : break
        cap.release()
        cv2.destroyAllWindows()

    else:
        img = cv2.imread('image.jpg')
        img = cv2.resize(img, (0, 0), None, 1.5, 1.5)
        df_sort = utils.util(img)
        cv2.waitKey(0)

    return df_sort

##############################################################################

df_sort = video(cam=True)
print(df_sort)