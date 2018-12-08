# Public packages
import os
import cv2
import glob

# Private packages
import face_detection as FACED
import face_recognition as FACER
import emoji_cover as GRAFT

# Parameters
K = 10
threshold = 1

# Paths
ALBUM_PATH = 'album/'

def main(K,threshold):
	# Step1 - calculate the eigenfaces
	...

	# Step2 - get cropped faces from all photos
	# To simplify, here just take one photo
	faces_set = {}
	faces_parent = []
	for photo_path in glob.glob(ALBUM_PATH+'*.jpg'):
		photo_name = os.path.splitext(os.path.basename(photo_path))[0]
		gray,faces = FACE.detect_dlib(photo_path,photo_name)
		face_parent = face_parent+[photo_name]*len(faces)
		faces_set[photo_name] = faces

	# Step3 - get all faces of one same person
	...
	weight_mat = get_faces_weight(cropped_faces,eigenfaces)
	dist_mat = get_dist_matrix(weight_mat)
	face_map = get_faces_from_same_person(0,dist_mat,threshold,face_parent)	# first face as sample

	# Step4 - graft emoji on same person's face
	for photo_path in glob.glob(ALBUM_PATH+'*.jpg'):
		photo_name = os.path.splitext(os.path.basename(photo_path))[0]
		img = cv2.imread(photo_path)
		faces = faces_set[photo_name]
		faces_info = get_faces_info(faces,face_map[photo_name])
		result = GRAFT.graft_emoji(img_path,faces_info)
		plt.imshow(result)
		plt.show()

def get_faces_info(faces,face_map):
	'''
	Get the face_info of those special faces
	'''
	faces_info = []
	labels = []
	count = 0
	for face in faces:
		if face_map[count] == 0:
			continue
		face_info = []
		face_info[0:2] = (FACE.get_face_info(gray,face))
		face_path = 'crop_faces/'+str(count)+'.jpg'
		face_img = cv2.imread(face_path)
		label = FACE.emotion_recognition(face_img)
		labels.append(label)
		face_info.append(label)
		count = count+1
		faces_info.append(face_info)
	return faces_info

if(__name__=="__main__"):
    main(K,threshold)