# CITATION
# for capturing landmarks real time on video
# José, Italo (2018) facial-landmarks-recognition [https://github.com/italojs/facial-landmarks-recognition/blob/master/main.py]
# detecting blinks skeleton code
# Rosebrock, Adrian (2017) Eye blink detection with OpenCV, Python, and dlib [https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/#pyis-cta-modal]
# /CITATION

import imutils
from imutils import face_utils
import dlib
import cv2
import ratios
import chime
import video_processing

class VideoRecorder():
    # constants
    # when EAR goes below then above this, a blink will be registered
    EAR_THRESHOLD =  0.2
    MOUTH_THRESHOLD = 0.75
    # total consecutive frames with EAR below the threshold necessary for a blink to be registered
    EAR_CONSEC_FRAMES = 2
    # for eyes closed too long
    # EAR_CONSEC_FRAMES = 75

    # class variables
    ear_frames_total = 0
    total_blinks = 0
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    left_start, left_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_start, right_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
    
    def video_processing(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        p = "shape_predictor_68_face_landmarks.dat"
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)

        # this line necessary when it wasnt a fxn
        # cap = cv2.VideoCapture(0)

        while True:
            _, image = self.capture.read()

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
            
                # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # extract coordinates and calculate EAR
                left_eye = shape[self.left_start:self.left_end]
                right_eye = shape[self.right_start:self.right_end]
                left_EAR = ratios.eye_aspect_ratio(left_eye)
                right_EAR = ratios.eye_aspect_ratio(right_eye)

                #extract coordinates for mouth and calculate MAR
                mouth = shape[self.mouth_start:self.mouth_end]

                # average EAR, we are assuming both eyes are blinking at the same time
                both_EAR = (left_EAR + right_EAR)/2.0

                # get MAR calculation
                mar = ratios.mouth_aspect_ratio(mouth)

                # if below threshold, we are entering a blink, add to number of frames this occurs
                if both_EAR < self.EAR_THRESHOLD:
                    self.ear_frames_total += 1
                else:
                    if self.ear_frames_total >= self.EAR_CONSEC_FRAMES:
                        self.total_blinks += 1
                    # for extended eyes closed/sleep
                    # if EAR_FRAMES_TOTAL >= EAR_CONSEC_FRAMES:
                        # chime.warning()
                    # reset our frames to 0 for another count
                    self.ear_frames_total = 0
                
                cv2.putText(image, "EAR: {:.2f}".format(both_EAR), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(image, "BLINKS: {}".format(self.total_blinks), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(image, "To stop, press q on your keyboard.", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", image)

            # video ends if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    def __del__(self):
        # clean up
        self.capture.release()
        cv2.destroyAllWindows()
