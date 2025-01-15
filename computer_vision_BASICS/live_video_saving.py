import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

four = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('car.mp4', four, 20.0, (width, height))

while cap.isOpened():
    rep, frame = cap.read()

    if rep == True:
        output.write(frame)

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
output.release()
cv2.destroyAllWindows()