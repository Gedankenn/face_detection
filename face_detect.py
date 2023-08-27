import cv2
import numpy as np
import dlib
# import face_recognition

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Initialize face detector (using dlib's pre-trained model)
face_detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the face detector
    faces = face_detector(gray)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Face Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
