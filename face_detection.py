import os
import cv2
import numpy
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt
import facenet.src.align.detect_face as detect_face

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


def box(rects, img, path):
    '''
    Mark detected faces and ranges
    '''
    # Get target file's base name
    base = os.path.splitext(os.path.basename(path))[0]

    # Add face boxes
    idx = 0
    for x1, y1, x2, y2 in rects:
        sub_img = img[y1 : y2, x1 : x2]
        cv2.imwrite('detected_faces/' + base + '_' + str(idx) + '.jpg', sub_img)
        cv2.imwrite('detected_faces/' + base + '_' + str(idx) + '.jpg', sub_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        idx = idx + 1

    cv2.imwrite('detected_cluster.jpg', img)


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


