import cv2  # Import OpenCV

# Load the pre-trained Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Open a video file or use a video capture device (0 for webcam)
video = cv2.VideoCapture('car.mp4')  # Replace with '0' for webcam

while True:
    # Read each frame from the video
    ret, frame = video.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert the frame to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
print("Video capture released and windows closed.")
