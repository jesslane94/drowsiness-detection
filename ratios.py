from scipy.spatial import distance

# close to 0 means the eye is blinking
# formula from: Soukupova, Tereza, and Jan Cech. Real-Time Eye Blink Detection Using Facial Landmarks, 2016, vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf. 
def eye_aspect_ratio(eye):
	# compute euclidean distances between the vertical eye landmarks (x, y)-coordinates
	v_distance1 = distance.euclidean(eye[1], eye[5])
	v_distance2 = distance.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal eye landmarks (x, y)-coordinates
	h_distance = distance.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (v_distance1 + v_distance2) / (2.0 * h_distance)
	return ear

# above idea can be extrapolated to the mouth
def mouth_aspect_ratio(mouth):
	# Vertical distance
	v_distance1 = distance.euclidean(mouth[2], mouth[10])  # 51, 59
	v_distance2 = distance.euclidean(mouth[4], mouth[8]) # 53, 57
	# Horizontal distance
	h_distance = distance.euclidean(mouth[0], mouth[6])  # 49, 55
	mar = (v_distance1 + v_distance2) / (2.0 * h_distance)
	return mar




