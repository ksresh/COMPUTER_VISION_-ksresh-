import cv2
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) == ord('q'):
        print("Exiting... 'q' key pressed.")
        break

camera.release()
cv2.destroyAllWindows()
print("Camera closed and windows closed.")
