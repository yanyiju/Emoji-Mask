import numpy as np
import imutils
import os
import cv2

# Folder of stored emoji png
EMOJI_FILE_PATH = "emoji_png/"

def graft_emoji(photo_path, faces):
	'''
	Graft the responding emoji onto the given photo based on face infomation 
	including face orientation (degree), face radius (ratio), face center, emotion label
	INPUT:
	photo - the path of target photo 
	faces - information of faces in the photo 
			list of list [orientation, radius, center, label]
			(can be optimized into numpy matrix)
	OUTPUT:
	result - the processed photo which labeled emojis have been added
	'''
	''' load photo '''
	photo = cv2.imread(photo_path,cv2.IMREAD_UNCHANGED)
	height,width,rgb = photo.shape
	''' grafting process '''
	result = photo
	for face in faces:
		emoji = cv2.imread(get_emoji_file(face[3]),cv2.IMREAD_UNCHANGED)
		rows,cols,rgba =  emoji.shape
		''' Get parameters for transforming '''
		angle = face[0]
		scale = 2*face[1]/rows
		''' Transform the emoji '''
		emoji_trans = rotate_img(emoji,angle)
		emoji_trans = scale_img(emoji_trans,scale)
		''' Graft '''
		result = overlay(result,emoji_trans,face[2])
	return result

def overlay(photo, emoji, pos):
	'''
	Overlay the emoji png onto the photo
	INPUT:
	photo - pre-processed photo (3 channels: RGB)
	emoji - the emoji ready to be placed (4 channels: RGBA)
	pos - the position of the emoji on photo
	OUTPUT:
	result - the intergrated photo
	'''
	result = photo
	''' Determine overlay range '''
	height,width = emoji.shape[0:2]
	(cx, cy) = pos
	(x1, y1) = (cx-int(height/2),cy-int(width/2))
	(x2, y2) = (cx+int(height/2),cy+int(width/2))
	''' Overlay '''
	for i in range(x2-x1+1):
		for j in range(y2-y1+1):
			''' Judge alpha channel '''
			if emoji[i,j,3] != 0:
				result[i,j] = emoji[i,j][0:3]
	return result

def rotate_img(img, angle):
	'''
	Rotate the emoji png image with the given angle
	INPUT:
	emoji - the emoji image (read by cv2) ready for rotation
	angle - the rotation angle
	OUTPUT: the rotated emoji image
	'''
	return imutils.rotate_bound(img,angle)

def scale_img(img, scale):
	'''
	Zoom in/out the emoji png with the given scale
	INPUT:
	emoji - the emoji image (read by cv2) ready for rotation
	scale - the scale parameter
	OUTPUT: the scaled emoji image
	'''
	x,y = round(img.shape[0]*scale),round(img.shape[1]*scale)
	return cv2.resize(img,(x,y),interpolation=cv2.INTER_CUBIC)

def get_emoji_file(label):
	'''
	Get the emoji png file name according to the emotion label
	INPUT:
	label - emotion label
	OUTPUT:
	filename - the corresponding emoji file name/path
	'''
	if label is 0:
		file = EMOJI_FILE_PATH+"neutral.png"
	elif label is 1:
		file = EMOJI_FILE_PATH+"angry.png"
	elif label is 2:
		file = EMOJI_FILE_PATH+"contempt.png"
	elif label is 3:
		file = EMOJI_FILE_PATH+"disgust.png"
	elif label is 4:
		file = EMOJI_FILE_PATH+"fear.png"
	elif label is 5:
		file = EMOJI_FILE_PATH+"happy.png"
	elif label is 6:
		file = EMOJI_FILE_PATH+"sad.png"
	elif label is 7:
		file = EMOJI_FILE_PATH+"surprise.png"
	return file

def check(img):
	'''
	Used for check how the image looks like
	'''
	cv2.imshow("temp",img)
	cv2.waitKey(0)
