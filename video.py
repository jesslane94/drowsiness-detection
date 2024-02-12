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

class VideoRecorder():
    # CLASS CONSTANTS
    # when EAR goes below then above this, a blink will be registered
    EAR_THRESHOLD =  0.3
    # close to one, yawn occuring
    MOUTH_THRESHOLD = 0.6
    # eyes close for too long
    EAR_CONSEC_FRAMES = 80
    # mouth open for a specific time 
    MOUTH_CONSEC_FRAMES = 80
    
    # grab the indexes of the facial landmarks for the left and right eye, respectively. also beginning/end of mouth corners.
    left_start, left_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_start, right_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    # CLASS VARIABLES
    # num frames at which the eyes are closed/mouth is open respectively
    ear_frames_total = 0
    mouth_frames_total = 0
    # keep count of times a "sleep" occurs or a yawn
    total_drowsiness = 0
    total_yawns = 0

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
    
    def video_processing(self):
        # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
        p = "shape_predictor_68_face_landmarks.dat"
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)

        while True:
            _, image = self.capture.read()
            
            # resize to 600 pixels and convert to greyscale
            image = imutils.resize(image, width=600)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # detect faces in the grayscale image
            rects = detector(gray, 0)
            
            # loop over the face detections
            for (i,rect) in enumerate(rects):
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
                # average EAR, we are assuming both eyes are blinking at the same time
                both_EAR = (left_EAR + right_EAR)/2.0

                #extract coordinates for mouth and calculate MAR
                mouth = shape[self.mouth_start:self.mouth_end]
                # get MAR calculation
                mar = ratios.mouth_aspect_ratio(mouth)

                # if below threshold, we are entering a blink, add to number of frames this occurs
                if both_EAR < self.EAR_THRESHOLD:
                    VideoRecorder.ear_frames_total += 1
                    if VideoRecorder.ear_frames_total >= self.EAR_CONSEC_FRAMES:
                        VideoRecorder.total_drowsiness += 1
                        chime.warning()
                        VideoRecorder.ear_frames_total = 0
                else:
                    #reset detected eye closed frames to 0
                    VideoRecorder.ear_frames_total = 0
                    
                # if mar gets past the threshold, we are entering a yawn. add to number of frames this occurs
                if mar > self.MOUTH_THRESHOLD:
                    VideoRecorder.mouth_frames_total += 1
                    if VideoRecorder.mouth_frames_total >= self.MOUTH_CONSEC_FRAMES:
                        VideoRecorder.total_yawns += 1
                        chime.warning()
                        VideoRecorder.mouth_frames_total = 0
                else:
                    #reset detected mouth open frames to 0
                    VideoRecorder.mouth_frames_total = 0

                cv2.putText(image, "TOTAL DROWSINESS: {}".format(VideoRecorder.total_drowsiness), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.putText(image, "TOTAL YAWNS: {}".format(VideoRecorder.total_yawns), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.putText(image, "To stop, press q on your keyboard.", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            
            # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", image)
            # video ends if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    # return dictionary of final counts for drowsiness and yawns
    def get_totals(self):
        totals = {}
        totals["total_drowsiness"] = VideoRecorder.total_drowsiness
        totals["total_yawns"] = VideoRecorder.total_yawns
        return totals
    
    def __del__(self):
        # clean up
        cv2.destroyAllWindows()
        self.capture.release()
        
