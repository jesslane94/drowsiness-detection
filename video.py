# CITATION
# José, Italo (2018) facial-landmarks-recognition [https://github.com/italojs/facial-landmarks-recognition/blob/master/main.py]
# /CITATION

import imutils
from imutils import face_utils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    _, image = cap.read()

    # initial test code for video 
    """ cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break """
    
    # resize to 400 pixels and convert to greyscale
    image = imutils.resize(image, width=400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)

    # video ends if q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# clean up
cap.release()
cv2.destroyAllWindows()
