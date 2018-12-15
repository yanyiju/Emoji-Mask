# Public packages
import os
import shutil
import cv2
import glob
import json
import imutils
import numpy as np
import matplotlib.pyplot as plt

# Private packages
import face_detection as FACED
import face_recognition as FACER
import emoji_cover as GRAFT

# Paths
ALBUM_PATH = "album/"
CROP_FACES_PATH = "crop_faces/"
PARAMETERS_PATH = "./parameters.json"

def main():
	# Read parameters
	with open(PARAMETERS_PATH, 'r') as f:
		parameters = json.load(f)
		threshold = parameters["threshold"]
		sample_num = parameters["sample_num"]

	# Clear history results
	if os.path.exists(CROP_FACES_PATH):
		shutil.rmtree(CROP_FACES_PATH)
		print(CROP_FACES_PATH+' is deleted.')
		mkdir(CROP_FACES_PATH)

	# Step1 - get cropped faces from all photos
	# To simplify, here just take one photo
	faces_num = 0
	faces_id = {}
	faces_set = {}
	faces_parent = []
	feature_mat = np.zeros((0,6))
	for photo_path in glob.glob(ALBUM_PATH+'*.jpg'):
		photo_name = os.path.splitext(os.path.basename(photo_path))[0]
		print(photo_name)
		mkdir(CROP_FACES_PATH+photo_name)
		gray,faces = FACED.detect_dlib(photo_path,photo_name+'/')
		for face in faces:
			faces_id[face] = faces_num
			faces_num = faces_num+1
			feature_mat = np.vstack((feature_mat,FACED.get_face_features(gray,face)))
		faces_parent = faces_parent+[photo_name]*len(faces)
		faces_set[photo_name] = faces

	# Step2 - get all faces of one same person
	dist_mat = FACER.get_dist_matrix(feature_mat)
	face_map = FACER.get_faces_from_same_person(sample_num,dist_mat,threshold,faces_parent)
	print(face_map)

	# Step3 - graft emoji on same person's face
	for photo_path in glob.glob(ALBUM_PATH+'*.jpg'):
		photo_name = os.path.splitext(os.path.basename(photo_path))[0]
		img = cv2.imread(photo_path)
		faces = faces_set[photo_name]
		if photo_name in face_map:
			photo = cv2.imread(photo_path)
			photo = imutils.resize(photo, width=800)
			cv2.imwrite('resize.png',photo)
			gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
			faces_info = get_faces_info(faces,face_map[photo_name],faces_id,gray,photo_name)
			img_path = 'resize.png'
			result = GRAFT.graft_emoji(img_path,faces_info)
			plt.imshow(result)
			plt.show()

def get_faces_info(faces,face_map,faces_id,gray,photo_name):
	'''
	Get the face_info of those special faces
	'''
	faces_info = []
	labels = []
	count = 0
	for face in faces:
		if faces_id[face] not in face_map:
			continue
		face_info = []
		print(face)
		face_info[0:2] = (FACED.get_face_info(gray,face))
		face_path = CROP_FACES_PATH+photo_name+'/'+str(count)+'.jpg'
		face_img = cv2.imread(face_path)
		label = FACED.emotion_recognition(face_img)
		labels.append(label)
		face_info.append(label)
		count = count+1
		faces_info.append(face_info)
	return faces_info

def mkdir(path):
	'''
	Make directory
	'''
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
		print('----New dir '+path+' built----')
	else:
		print('----Dir '+path+' already exists----')


if(__name__=="__main__"):
    main()