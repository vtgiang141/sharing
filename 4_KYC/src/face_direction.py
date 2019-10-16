import dlib
import math
import imutils
from imutils import face_utils
import cv2
import numpy as np


# Position of points in 68 points facial landmark
RIGHT_EYE = [36, 37, 38, 39, 40, 41]
LEFT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
detector = dlib.get_frontal_face_detector()
facial_landmark_data = "../Models/model_dlib/shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(facial_landmark_data)
except:
    predictor = dlib.shape_predictor("../" + facial_landmark_data)
class DetectDirection(object):
    
    def __init__(self, input_frame):
        self.frame = input_frame
        
    def get_center(self, pointList, map68FacialLandmark, dataInMap = True):
        ### Calculate center point in (x,y)-coordiante of pointList which is extracted from 'shape' variable
        sumPoint = [0, 0]
        for point in pointList:
            if dataInMap:
                sumPoint += map68FacialLandmark[point]
            else:
                sumPoint += point
        centerPoint = sumPoint // len(pointList)
        return centerPoint

    def get_distance_between_2_point(self, point1, point2, dataInMap = True):
        ### Calculate distance between point1 to point2 in pixel, the 'dataInMap' flag is in 68 point facial landmark 
        vector = point2 - point1
        square = 0
        for dimention in vector:
            square += dimention*dimention
        return math.sqrt(square)

    ## Return the map 
    def get_facial_landmark_map(self):
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame1 = imutils.resize(self.frame, width=400)
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rect = detector(gray, 0)
        # print(type(gray))
        # print(gray)
        # print(type(rect))
        # print(rect)
        if len(rect) == 1:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect[0])
            shape = face_utils.shape_to_np(shape)
            return [True, shape]
        else:
            return False

    def get_pan_angle(self):
        landmark = self.get_facial_landmark_map()
        if isinstance(landmark, list):
            shape = landmark[1]
            ### Calculate points for face direction detection (pan angle)
            leftEyeCenterPosition = self.get_center(LEFT_EYE, shape)
            rightEyeCenterPosition = self.get_center(RIGHT_EYE, shape)

            ### Distance from left/right pupil to left/right face edge
            Lx = self.get_distance_between_2_point(leftEyeCenterPosition, shape[1], False)
            Rx = self.get_distance_between_2_point(rightEyeCenterPosition, shape[16], False)
            ### Face pan direction angle
            faceDirectionAngel = np.arcsin((Rx-Lx)/(Rx+Lx)) * 180 / 3.1415
            return faceDirectionAngel
        else:
            return False

    def get_rot_angle(self):
        landmark = self.get_facial_landmark_map()
        if isinstance(landmark, list):
            shape = landmark[1]
            ### Calculate points in face 
            leftEyeCenterPosition = self.get_center(LEFT_EYE, shape)
            rightEyeCenterPosition = self.get_center(RIGHT_EYE, shape)
            ### Calculate points for face direction detection (rot angle)
            mouthCenterPosition = self.get_center(MOUTH, shape)
            centerOfPupils = self.get_center([leftEyeCenterPosition, rightEyeCenterPosition], shape, False)
            rotVector = centerOfPupils - mouthCenterPosition
            rotAngle = np.arctan(rotVector[0]/rotVector[1]) * 180 / 3.1415
            return rotAngle
        else:
            return False
