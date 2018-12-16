############################################################
##  
##  Private package 
##  Function: Detecting faces
##
############################################################
import argparse
import imutils
import dlib
import cv2
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import facenet.src.align.detect_face as detect_face
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
from scipy import misc

CROP_FACES_PATH = "crop_faces/"

''' find_face_solution 1 '''

# Custormized, load preset for detect_cv2
CASCADE = "haarcascade_frontalface_alt.xml"
FACE_DETECTION = cv2.CascadeClassifier(CASCADE)

def detect_cv2(path):
    '''
    Face detection using opencv package
    '''
    # Face detection
    img = cv2.imread(path)
    faces = FACE_DETECTION.detectMultiScale(img, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (20,20))
    # No face detected
    if len(faces) == 0:
        return [], img
    faces[:, 2:] += faces[:, :2]
    return faces, img

def box_cv2(faces, img, path):
    '''
    Mark detected faces and ranges
    '''
    # Get target file's base name
    base = os.path.splitext(os.path.basename(path))[0]
    # Add face boxes
    idx = 0
    for x1, y1, x2, y2 in faces:
        sub_img = img[y1 : y2, x1 : x2]
        cv2.imwrite('detected_faces/' + base + '_' + str(idx) + '.jpg', sub_img)
        cv2.imwrite('detected_faces/' + base + '_' + str(idx) + '.jpg', sub_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        idx = idx + 1
    cv2.imwrite('detected_cluster.jpg', img)

''' find_face_solution 2 '''

# Custormized, load preset for detect_mtcnn
minsize = 20                                    # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]                   # three steps's threshold
factor = 0.709                                  # scale factor
gpu_memory_fraction=1.0
 
def detect_mtcnn(path):
    '''
    Face detection using facenet & tensorflow package
    '''
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)         
     
    img = misc.imread(path)            
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    num_faces = bounding_boxes.shape[0]
    print('////////////{} faces founded////////////'.format(nrof_faces)) 
    print(bounding_boxes)
     
    crop_faces=[]
    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        print(face_position[0:4])
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=img[face_position[1]:face_position[3],
                 face_position[0]:face_position[2],]
        
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
        print(crop.shape)
        crop_faces.append(crop)
        plt.imshow(crop)
        plt.show()
        
    plt.imshow(img)
    plt.show()

''' find_face_solution 3 '''

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
DLIB_DETECTOR = dlib.get_frontal_face_detector()
DLIB_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_dlib(path,name):
    '''
    Face detection using dlib face detector
    INPUT:
    path - the path of the photo ready for process
    OUTPUT:
    gray - the gray scaled image, needed by func get_face_center()
    faces - the detected faces
    '''
    image = cv2.imread(path)
    image = imutils.resize(image, width=800)
    cv2.imwrite('resize.png',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = DLIB_DETECTOR(gray, 2)
    print(faces)
    count = 0
    for face in faces:
        # Save the cropped face
        face_name = CROP_FACES_PATH+name+str(count)+'.jpg'
        get_cropped_face(face,face_name,image)
        count = count+1
    return gray,faces

def get_face_info(gray,face):
    '''
    This function follows func detect_dlib()
    Return the center pos/radius/2D orientation of the given face
    INPUT:
    gray - the gray scaled image used for prediction
    face - the given face
    OUTPUT:
    angle - the orientation of the face in 2D 
    radius - the radius of the face range, unit in pixels
    center - the position/center of the face, set as the middle point between eyes
    '''
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    shape = DLIB_PREDICTOR(gray, face)
    shape = shape_to_np(shape)

    # Left eye
    avg_x_left,avg_y_left = get_feature_pos("left_eye",shape)
    cv2.circle(gray,(int(avg_x_left),int(avg_y_left)),1,(128,128,128),-1)

    # Right eye
    avg_x_right,avg_y_right = get_feature_pos("right_eye",shape)
    cv2.circle(gray,(int(avg_x_right),int(avg_y_right)),1,(128,128,128),-1)

    # Calculate the angle (2D orientation)
    angle = np.arctan((avg_y_right-avg_y_left)/(avg_x_right-avg_x_left))
    angle = angle/np.pi*180
    # Calculate the radius
    width = (face.height()+face.width())/2.0
    # Calulate center
    center = (int((avg_x_left+avg_x_right)/2),int((avg_y_left+avg_y_right)/2))

    return angle,width,center


def get_face_features(gray,face):
    '''
    Used to judge if two faces are the same person
    INPUT:
    gray - the gray scaled image used for prediction
    face - the given face
    OUTPUT:
    features - a vector recording the distances between face features
    Here I take the person's nose as the reference point, thus in the 
    following expression dist[<feature>] means the distance between 
    his/her nose and the feature point.

    The order of features np array:
        [dist["mouth"], dist["right_eyebrow"], dist["left_eyebrow"], 
            dist["right_eye"], dist["left_eye"], dist["jaw"]]
    '''
    shape = DLIB_PREDICTOR(gray, face)
    shape = shape_to_np(shape)

    positions = []
    nose = get_feature_pos("nose",shape)
    positions.append(get_feature_pos("mouth",shape))
    positions.append(get_feature_pos("right_eyebrow",shape))
    positions.append(get_feature_pos("left_eyebrow",shape))
    positions.append(get_feature_pos("right_eye",shape))
    positions.append(get_feature_pos("left_eye",shape))
    positions.append(get_feature_pos("jaw",shape))

    # Get the vector
    features = np.zeros((1,6))
    for i in range(6):
        features[0,i] = get_distance(nose,positions[i])
    # Normalize
    features = features/np.sqrt(np.sum(features**2))
    
    return features


def get_feature_pos(feature,shape):
    '''
    Get the average value/pos of the points for the feature
    INPUT:
    feature - a string naming the feature, like "mouth"
    shape - a numpy array including all points
    OUTPUT:
    position - the position of the feature
    '''
    (start, end) = FACIAL_LANDMARKS_IDXS[feature]
    point_set = shape[start:end]

    # get the geometry center/average pos value
    sum_x = 0
    sum_y = 0
    for (x, y) in point_set:
        sum_x = sum_x + x
        sum_y = sum_y + y
    if end-start != 0:
        avg_x = sum_x/(end-start)
        avg_y = sum_y/(end-start)
    else:
        avg_x = np.inf
        avg_y = np.inf
    return avg_x,avg_y


def get_distance(f1,f2):
    '''
    Get the distance between two features on face.
    INPUT:
    f1 - position of face feature 1
    f2 - position of face feature 2
    OUTPUT:
    dist - the distance
    '''
    x1,y1 = f1
    x2,y2 = f2
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


def get_cropped_face(face,face_name,img):
    '''
    Get and save the cropped face.
    '''
    print(face_name)
    y1,y2,x1,x2 = get_face_range(face)
    print(x1,x2,y1,y2)
    face_img = img[x1:x2,y1:y2]
    plt.imshow(face_img)
    face_img = cv2.resize(face_img,(96,96),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(face_name,face_img)

def add_box_text(faces,labels,img):
    '''
    Mark detected faces and emotions
    '''
    img = imutils.resize(img, width=800)
    idx = 0
    for face in faces:
        x1,x2,y1,y2 = get_face_range(face)
        # Add face box
        cv2.rectangle(img,(x1,y1),(x2,y2),(127, 255, 0),2)
        # Add face emotion text
        cv2.putText(img,labels[idx],(x1,y1-15),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0))
        idx = idx+1
    return img

def get_face_range(face):
    '''
    Get the face range from dlib.rectangle object
    '''
    l = face.left()
    r = face.right()
    t = face.top()
    b = face.bottom()
    if l < 0:
        l = 0
    if r < 0:
        r = 0
    if t < 0:
        t = 0
    if b < 0:
        b = 0     
    return l,r,t,b