
import os
import face_recognition
import cv2
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load and encode known faces (replace with your own images)
image_path = "faces/"
known_names = ["fabio"]

for name in known_names:
    image = face_recognition.load_image_file(f"{image_path}{name}.jpg")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the frameq
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]

        # Draw a rectangle and name label on the frame
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
