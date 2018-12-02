import numpy as np
import os
import cv2

# from PIL import image

# Folder of stored emoji png
EMOJI_FILE_PATH = "emoji/"

def graft_emoji(photo_path, faces):
	'''
	Graft the responding emoji onto the given photo based on face infomation 
	including face orientation (degree), face radius (ratio), face center, emotion label
	input:
	photo - the path of target photo 
	faces - information of faces in the photo 
			list of list [orientation, radius, center, label]
			(can be optimized into numpy matrix)
	output:
	result - the processed photo which labeled emojis have been added
	'''
	# load photo
	photo = cv2.imread(photo_path,cv2.IMREAD_UNCHANGED)
	height,width,rgb = photo.shape
	# grafting process
	result = photo
	for face in faces:
		emoji = cv2.imread(get_emoji_file(face[3]),cv2.IMREAD_UNCHANGED)
		rows,cols,rgba =  emoji.shape

		# Transform the emoji
		scale = face[1]*width/rows
		(size_x,size_y) = (int(rows*scale),int(cols*scale)) 
		trans_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),face[0],scale)
		emoji_trans = cv2.warpAffine(emoji,trans_matrix,(rows,cols))

		# Graft
		(px,py) = face[2]
		(px1,py1) = (int(px-size_x/2),int(py-size_y/2))
		(px2,py2) = (int(px+size_x/2),int(py+size_y/2))
		(ex1,ey1) = (int(rows/2-size_x/2),int(cols/2-size_y/2))
		(ex2,ey2) = (int(rows/2+size_x/2),int(cols/2+size_y/2))
		result[px1:px2,py1:py2] = emoji_trans[ex1:ex2,ey1:ey2,0:3]

	return result

def get_emoji_file(label):
	'''
	Get the emoji png file name according to the emotion label
	input:
	label - emotion label
	output:
	filename - the corresponding emoji file name/path
	'''
	file = EMOJI_FILE_PATH+str(label)+".png"
	return file

def check(img):
	'''
	Used for check how the image looks like
	'''
	cv2.imshow("temp",img)
	cv2.waitKey(0)
