import cv2
from keras.models import load_model
import numpy as np
import os
import imageio


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

emotion_list = ['AN', 'DI', 'FE', 'HA', 'SA', 'SU', 'NE']

'''take in already cut face img'''
def emotion_recognition(img):
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_list[emotion_label_arg]
    return (emotion)

# img=cv2.imread("friend1.jpg_0.jpg")

test_path = "../jaffe"
test_file = os.listdir(test_path)
correct_count = 0
error_count = 0
for i in range(len(test_file)):
    img = imageio.imread(test_path + '/' + test_file[i])
    if emotion_recognition(img) == test_file[i][3:5]:
        correct_count += 1
    else:
        error_count += 1
print(correct_count / (correct_count + error_count))
# print(emotion_recognition(img))
