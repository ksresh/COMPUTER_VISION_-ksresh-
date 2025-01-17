import cv2
import numpy as np

img = np.zeros((512,512,3), dtype = np.uint8) # (np.uint8) to make as unsinged 8 bit integer
img[ : ] = [255,0,0]  # making all image pixels to blue color

cv2.line(img, (0,0),(img.shape[1]//2,img.shape[0]//2), (0,0,255),3)
cv2.line(img, (img.shape[1]//2,img.shape[0]//2), (img.shape[1],0), (0,255,0),3)

cv2.rectangle(img, (200,200), (img.shape[1]-200, img.shape[0]-200), (255,0,255), cv2.FILLED)  # filling the content
cv2.circle(img, (img.shape[1]//2,img.shape[0]//2), 200, (255,255,255), 3)

cv2.putText(img,
            'DESAIAH',
            ((img.shape[1]//2)-140,(img.shape[0]//2)+120),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (150,150,150),
            3)

cv2.imshow('img', img)

cv2.waitKey(0)