import cv2
import face_recognition
import numpy as np
import time

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

# Define known face encodings and names
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

# Attendance dictionary
attendance = {name: 'Absent' for name in known_face_names}

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Threshold for blinking detection
blink_threshold = 0.25  # Adjusted blink threshold

blinking_detected = False  # Flag for blinking detection
recognized = False  # Flag for face recognition
recognition_timer = None  # Timer for face recognition message
face_in_frame = False  # Flag to track if face is in frame

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Check if frame is empty
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    # Find all the faces and face landmarks in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Check for blinking
    for (top, right, bottom, left), landmarks in zip(face_locations, face_recognition.face_landmarks(frame, face_locations)):
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            left_eye_width = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[3]))
            right_eye_width = np.linalg.norm(np.array(right_eye[0]) - np.array(right_eye[3]))
            eye_aspect_ratio = (np.linalg.norm(np.array(left_eye[1]) - np.array(left_eye[5])) + np.linalg.norm(np.array(left_eye[2]) - np.array(left_eye[4]))) / (2 * left_eye_width)
            eye_aspect_ratio += (np.linalg.norm(np.array(right_eye[1]) - np.array(right_eye[5])) + np.linalg.norm(np.array(right_eye[2]) - np.array(right_eye[4]))) / (2 * right_eye_width)
            eye_aspect_ratio /= 2
            if eye_aspect_ratio < blink_threshold:
                blinking_detected = True
                break

    # Reset recognition if no face is detected
    if not face_locations:
        recognized = False
        face_in_frame = False

    # Loop through each face found in the frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Change the name to the individual's name if blinking is detected and not already recognized
            if blinking_detected and not recognized and face_in_frame:
                recognized = True
                name = known_face_names[first_match_index]
                message = f"Face recognized. Welcome, {name}"
                cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                recognition_timer = time.time()
                # Update attendance
                attendance[name] = 'Present'

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if recognized:
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display blinking checklist or recognition message
    if recognized:
        if recognition_timer is not None and time.time() - recognition_timer < 5:
            cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            recognized = False
            blinking_detected = False
    elif face_in_frame:
        cv2.putText(frame, "Blink your eyes", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check if recognition message timer has expired
    if recognition_timer is not None and time.time() - recognition_timer >= 5:
        recognized = False
        recognition_timer = None

    # Reset blinking detection flag after recognition
    if recognized:
        blinking_detected = False

    # Check if face is in frame
    if face_locations:
        face_in_frame = True
    else:
        face_in_frame = False

    # Hit 'q' on the keyboard to quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Display attendance
print("Attendance:")
for name, status in attendance.items():
    print(f"{name}: {status}")
