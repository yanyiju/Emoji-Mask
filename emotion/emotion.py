import cv2
from keras.models import load_model
import numpy as np


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

'''take in already cut face img'''
def emotion_recognition(img):
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    return (emotion)

img=cv2.imread("friend1.jpg_0.jpg")
print(emotion_recognition(img))
