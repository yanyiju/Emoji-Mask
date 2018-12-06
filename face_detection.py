import argparse
import imutils
import dlib
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import facenet.src.align.detect_face as detect_face
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
from scipy import misc
from keras.models import load_model

CROP_FACES_PATH = "crop_faces/"
emotion_classifier = load_model('simple_CNN.530-0.65.hdf5')
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

''' <emotion.py> '''

def emotion_recognition(img):
    '''take in already cut face img'''
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    # emotion = emotion_labels[emotion_label_arg]
    return emotion_label_arg

''' <find_face_solution1.py> '''

# Custormized, load preset for detect_cv2
CASCADE = "haarcascade_frontalface_alt1.xml"
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
    # cv2.imwrite('detected_cluster.jpg', img)

''' <find_face_solution2.py> '''

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

''' <face_align.py> '''

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
DLIB_DETECTOR = dlib.get_frontal_face_detector()
DLIB_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_dlib(path):
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
    count = 0
    for face in faces:
        # Save the cropped face
        face_name = CROP_FACES_PATH+str(count)+'.jpg'
        face_img = get_cropped_face(face,face_name,image)
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

    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # Left eye
    sum_x_left = 0
    sum_y_left = 0
    count = 0
    for (x, y) in leftEyePts:
        sum_x_left = sum_x_left + x
        sum_y_left = sum_y_left + y
        count = count + 1
        # cv2.circle(gray,(x,y),1,(0,0,255),-1)
    avg_x_left = sum_x_left/count
    avg_y_left = sum_y_left/count
    cv2.circle(gray,(int(avg_x_left),int(avg_y_left)),1,(128,128,128),-1)
    
    # Right eye
    sum_x_right = 0
    sum_y_right = 0
    count = 0
    for (x, y) in rightEyePts:
        sum_x_right = sum_x_right + x
        sum_y_right = sum_y_right + y
        count = count + 1
        # cv2.circle(gray, (x, y), 1, (255, 255, 0), -1)
    avg_x_right = sum_x_right/count
    avg_y_right = sum_y_right/count
    cv2.circle(gray,(int(avg_x_right),int(avg_y_right)),1,(128,128,128),-1)

    # Calculate the angle (2D orientation)
    angle = np.arctan((avg_y_right-avg_y_left)/(avg_x_right-avg_x_left))
    angle = angle/np.pi*180
    # Calculate the radius
    radius = face.height()+face.width()
    # Calulate center
    center = (int((avg_x_left+avg_x_right)/2),int((avg_y_left+avg_y_right)/2))

    return angle,radius,center

def get_cropped_face(face,face_name,img):
    y1 = face.left()
    y2 = face.right()
    x1 = face.top()
    x2 = face.bottom()
    face_img = img[x1:x2,y1:y2]
    face_img = cv2.resize(face_img,(96,96),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(face_name,face_img)
    return face_img

def add_box_text(faces,labels,img):
    '''
    Mark detected faces and emotions
    '''
    img = imutils.resize(img, width=800)
    idx = 0
    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        # Add face box
        cv2.rectangle(img,(x1,y1),(x2,y2),(127, 255, 0),2)
        # Add face emotion text
        cv2.putText(img,emotion_labels[labels[idx]],(x1,y1-15),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0))
        idx = idx+1
    cv2.imwrite('box_text.png', img)
