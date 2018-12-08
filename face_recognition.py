############################################################
##  
##  Private package 
##  Function: Detecting faces' belonging (same person)
##
############################################################
import cv2
import numpy as np
from numpy import linalg as LA

def get_eigenfaces(images,K):
    n, length = images.shape
    average = np.zeros((1, length))

    # First, get the average face.
    for i in range(n):
        average = average + images[i].reshape((1, length))
    average = average/n

    # Then, get the diff images
    diff = np.zeros((n, length))
    for i in range(n):
        diff[i, :] = images[i].reshape((1, length)) - average

    # Lastly, get the covariance (X.T * X) and engenvectors
    # covariance = np.asmatrix(diff.T) * np.asmatrix(diff)
    covariance = np.cov(images, rowvar = False)
    eigenVal, eigenVec = LA.eig(covariance)
    # eigenVec = np.asmatrix(diff.transpose()) * eigenVec_c
    if K > length:
        K = length
    eigenfaces = eigenVec.T[:K]
    return eigenfaces,average

def get_faces_weight(images,eigenfaces):
	'''
	Get the weight for the cropped img on prepared eigen faces
	INPUT:
	images - the ready cropped images for all photoes
	eigenfaces -  the matrix provided in function get_eigenfaces()
	OUTPUT:
	weight - a numpy array including the weights
	'''
    K = np.size(eigenbasis, 0)
    n = np.size(images, 0)
    length = np.size(images, 1)
    weight_mat = np.zeros((n, K))
    for i in range(n):
        image = images[i].reshape((1, length))
        weight_mat[i, :] = np.dot(image, eigenbasis.T)
    return weight_mat

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

def get_dist_matrix(weight_mat):
	'''
	Get the distance matrix (nxn) for n cropped faces
	INPUT:
	weight_mat - nxK matrix
	OUTPUT:
	dist_mat - nxn matrix
	'''
	n,K = weight_mat.shape
	dist_mat = np.zeros(n,n)
	for i in range(n):
		for j in range((i+1):n):
			dist_mat[i,j] = get_face_distance(weight_mat[i],weight_mat[i])
			dist_mat[j,i] = weight_mat[i,j]
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
	face_map - an array of face number for same person
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


