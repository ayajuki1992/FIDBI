import cv2
import os

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Load Haar cascade classifier for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

ret, img = cam.read()
img = cv2.flip(img, 1) # flip video image horizontally
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     

    # Save the captured image into the datasets folder
    cv2.imwrite("dataset/" + str(face_id) + ".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('image', img)

# Press 'ESC' for exiting video
cv2.waitKey(0)

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()