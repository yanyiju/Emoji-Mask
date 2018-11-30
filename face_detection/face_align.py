# import the necessary packages
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import argparse
import imutils
import dlib
import cv2
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

# show the original input image and detect faces in the grayscale
# image
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("friend1.jpg")
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 2)

 
# show the original input image and detect faces in the grayscale
# image
# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    sum_x_left = 0
    sum_y_left = 0
    count = 0
    for (x, y) in leftEyePts:
        sum_x_left = sum_x_left + x
        sum_y_left = sum_y_left + y
        count = count + 1
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    avg_left_x = sum_x_left / count
    avg_left_y = sum_y_left / count
    cv2.circle(image, (int(avg_left_x), int(avg_left_y)), 1, (128, 128, 128), -1)

    
    sum_x_right = 0
    sum_y_right = 0
    count = 0
    for (x, y) in rightEyePts:
        sum_x_right = sum_x_right + x
        sum_y_right = sum_y_right + y
        count = count + 1
        cv2.circle(image, (x, y), 1, (255, 255, 0), -1)
    avg_right_x = sum_x_right / count
    avg_right_y = sum_y_right / count
    cv2.circle(image, (int(avg_right_x), int(avg_right_y)), 1, (128, 128, 128), -1)

    angle = np.arctan((avg_right_y - avg_left_y) / (avg_right_x - avg_left_x))
    angle = - angle / np.pi * 180
    print(angle)

cv2.imshow("Aligned", image)
cv2.waitKey(0)

