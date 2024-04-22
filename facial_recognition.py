#Attention! This code only works on MAC OS. If you are using Windows, you need to change the video_capture index to 0.
#Changing the index to 0 is only a MAYBE. Not yet tested on Windows. Otheriwse, you're out of luck.

#ALSO! You need to install OpenCV library to run this code.
#You can install it by running the following command in your terminal: pip install opencv-python

import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(1)
test  
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "FIDBI face_reg", video_frame
    )  # display the processed video frame with the name 'FIDBI face_reg'

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

