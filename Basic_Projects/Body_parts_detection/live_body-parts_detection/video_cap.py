import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model  # Importing load_model from TensorFlow

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 620)
cap.set(4, 360)

# Load the categories of body parts
DIRECTORY = r'Body_parts_detection'
CATEGORIES = os.listdir(DIRECTORY)
CATEGORIES = [cat for cat in CATEGORIES if cat != '.DS_Store']

# Load the trained model using TensorFlow's load_model
model1 = load_model('model_vgg16.h5')

while cap.isOpened():
    # Capture each frame from the webcam
    ref, img = cap.read()

    if img is not None:
        # Make a copy of the original frame
        img_copy = img.copy()

        # Preprocess the frame (resize and normalize)
        img_resized = cv2.resize(img, (256, 256))  # Resize to match model input size
        img_resized = img_resized / 255  # Normalize the image
        img_resized = np.expand_dims(img_resized, axis=0)  # Expand dims to match batch shape

        # Predict the class of the object in the image
        prediction = model1.predict(img_resized, verbose=0)
        predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
        predicted_class = CATEGORIES[predicted_class_index]  # Get the class name
        confidence = prediction[0][predicted_class_index] * 100  # Get the confidence score (accuracy)

        # Display the predicted class name and confidence on the frame
        cv2.putText(img_copy,
                    f'Object: {predicted_class}',  # Display object name
                    (50, 50),  # Position on the screen
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,  # Font size
                    (0, 0, 255),  # Text color (red)
                    2)  # Thickness of text

        cv2.putText(img_copy,
                    f'Confidence: {confidence:.2f}%',  # Display confidence
                    (50, 100),  # Position below the object name
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,  # Font size
                    (0, 255, 0),  # Text color (green)
                    2)  # Thickness of text

        # Show the image with the object name and confidence
        cv2.imshow('Real-Time Object Detection', img_copy)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
