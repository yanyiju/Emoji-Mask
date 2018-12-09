############################################################
##  
##  Private package 
##  Function: Detecting faces' belonging (same person)
##
############################################################
import cv2
import numpy as np
from numpy import linalg as LA

def get_face_distance(face1,face2):
	'''
	Get the distance between two faces' vectors based on the 
	eigen faces. Use norm2 to calculate.
	INPUT:
	face1 - a numpy array of the face (weight on eigen faces).
	face2 - the second face, also numpy array.
	OUTPUT:
	distance - a float/double describing the distance between 
	two faces.
	'''
	return LA.norm(face1-face2,ord=2)


def get_dist_matrix(feature_mat):
	'''
	Get the distance matrix (nxn) for n cropped faces
	INPUT:
	feature_mat - nx6 matrix since there are 6 features
	OUTPUT:
	dist_mat - nxn matrix
	'''
	n,K = feature_mat.shape		# K should be 6
	dist_mat = np.zeros((n,n))
	for i in range(n):
		for j in range((i+1),n):
			dist_mat[i,j] = get_face_distance(feature_mat[i],feature_mat[j])
			dist_mat[j,i] = dist_mat[i,j]
	return dist_mat


def get_faces_from_same_person(sample,dist_mat,threshold,face_parent):
	'''
	Get those faces from the same person as in sample
	INPUT:
	sample - a number in the range [0,n)
	dist_mat - a nxn matrix recording all face distances
	threshold - if distance < threshold, same person
	face_parent - recording which photo the faces are in
	OUTPUT:
	face_map - an set of face number for same person
	'''
	face_map = {}
	n = dist_mat.shape[0]
	for i in range(n):
		if dist_mat[sample,i] < threshold:
			parent = face_parent[i]
			if parent in face_map:
				face_map[parent].add(i)
			else:
				face_map[parent] = {i}
	return face_map


