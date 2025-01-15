import cv2

cap = cv2.VideoCapture(0)

width = int(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) or cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) or cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

four = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', four, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()  # Changed 'ref' to 'ret' for clarity

    if ret:  # Simplified condition
        output.write(frame)
        cv2.imshow('output_vid1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
output.release()
cv2.destroyAllWindows()

