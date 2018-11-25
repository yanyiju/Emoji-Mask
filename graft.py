import numpy as np
import os
import cv2

def graft_emoji(photo, faces):
	'''
	Graft the responding emoji onto the given photo based on face infomation 
	including face orientation (degree), face radius (ratio), emotion label
	input:
	photo - target photo 
	faces - information of faces in the photo 
			list of list [orientation, radius, label]
			(can be optimized into numpy matrix)
	output:
	result - the processed photo which labeled emojis have been added
	'''
	result = photo
	#TODO
	return result


def get_emoji_file(label):
	'''
	Get the emoji png file name according to the emotion label
	input:
	label - emotion label
	output:
	filename - the corresponding emoji file name/path
	'''
	file = "1.png"
	#TODO
	return file
