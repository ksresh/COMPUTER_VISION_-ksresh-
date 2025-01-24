import cv2
import numpy as np
import utils_1

######################################################
def video(cam):
    if cam:
        cap = cv2.VideoCapture(0)
        cap.set(3,780)
        cap.set(4,440)

        while cap.isOpened():
            ref, img = cap.read()
            if ref & img is not None:
                df_sort = utils_1.cont(img)
            if cv2.waitKey(1) & 0XFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread('image.jpg')
        if img is not None:
            df_sort = utils_1.cont(img)
        cv2.waitKey(0)

    return df_sort

df_sort = video(cam=True)
print(df_sort)

