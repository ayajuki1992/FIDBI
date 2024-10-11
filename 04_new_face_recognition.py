import face_recognition
import cv2

# Load the known images
image_maciej = face_recognition.load_image_file("dataset/User.4.jpg")
image_noraiz = face_recognition.load_image_file("dataset/User.3.jpg")
image_ismail = face_recognition.load_image_file("dataset/User.5.jpg")
image_sameer = face_recognition.load_image_file("dataset/User.2.jpg")
image_arjun = face_recognition.load_image_file("dataset/User.1.jpg")

# Get the face encodings for the known images
maciej_face_encoding = face_recognition.face_encodings(image_maciej)[0]
noraiz_face_encoding = face_recognition.face_encodings(image_noraiz)[0]
ismail_face_encoding = face_recognition.face_encodings(image_ismail)[0]
sameer_face_encoding = face_recognition.face_encodings(image_sameer)[0]
arjun_face_encoding = face_recognition.face_encodings(image_arjun)[0]

known_face_encodings = [
    maciej_face_encoding,
    noraiz_face_encoding,
    ismail_face_encoding,
    sameer_face_encoding,
    arjun_face_encoding,
    
]
known_face_names = [
    "Maciej",
    "Noraiz",
    'Ismail',
    'Sameer',
    'Arjun',
]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

