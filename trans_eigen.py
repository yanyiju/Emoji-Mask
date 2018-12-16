import os
import dlib
import numpy as np
import imageio
import cv2
from skimage import color
from helpers import shape_to_np
from PIL import Image
import matplotlib.pyplot as plt

DLIB_DETECTOR = dlib.get_frontal_face_detector()
DLIB_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

expression_path = "expression_set_img/"
output_path = "expression_set_img_points/"

img_row = 50
img_col = 50

def main():
	'''
	This file is used to transform/adjust the training image for emotion
	detection in order to get a better performance in eigenface algorithm.
	'''
	expression_files = os.listdir(expression_path)
	data_as_array = np.zeros((len(expression_files), img_row, img_col))
	for i in range(len(expression_files)):
		image = Image.open(expression_path + expression_files[i])
		image = np.array(image)
		gray,faces = detect(image)
		result = np.zeros((img_row, img_col))
		for face in faces:
			shape = DLIB_PREDICTOR(gray, face)
			shape = shape_to_np(shape)

			for point in shape:
				if point[0] >= 50:
					point[0] = 49
				if point[1] >= 50:
					point[1] = 49
				result[point[1],point[0]] = 255
		filename = expression_files[i]
		base = filename[0:(len(filename)-4)]
		imageio.imwrite(output_path+base+".png", result)
			


def detect(image):
	image = cv2.resize(image, (img_row, img_col))
	gray = color.rgb2gray(image)
	gray *= 255.0/gray.max()
	gray = np.uint8(gray)
	faces = DLIB_DETECTOR(gray, 2)
	return gray,faces

if(__name__=="__main__"):
    main()