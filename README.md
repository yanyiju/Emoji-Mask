# EECS442 Computer Vision Final Project: Emoji Mask

+ Overview
+ System Requirement
+ Running Instruction

## Overview
This project aims to cover emoji (or any specific mask) to people’s faces in a set of photo albums. Emoji Mask will identify people’s emotions in a photo and cover their faces by matching emojis with proper sizes and orientations. This project will not only serve as a fun method of photo processing but also provide a potential use in protecting personal privacy in social media (mask/unmask target’s face by learning the same person’s faces implemented).

### Face Detection
Since the related knowledge is not fully covered in classes, this part is finally realized through package dlib based on the referenced training model of tensorflow. (find face solution 3 is applied here: detect_dlib())

+ Note: *helpers.py* is referenced from [Github](https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py)

### Emotion Detection
Emotion detection is based on the eigen faces algorithm.
#### Expression Labels
~~~sh
0 = neutral
1 = angry
2 = contempt
3 = disgust
4 = fear 
5 = happy
6 = sadness
7 = surprise
~~~

### Emoji Grafting
During [Face Detection](./face_detection.py), we can get the key points' infomation of landmarks for each face. Based on the left eye and right eye information, we can roughly get the center of a person's face at the middle point between left and right eye and the orientation of the face from the slope of the line from left eye to the right one.

### Face Recognition (still in elementary stage)
This part is aimed to recognize one same person's faces across all given pictures given in the *./album*. Assuming that each face is a 2D plane and has a relatively small angle with the camera projective plane (or the face is very possibly nor detected), which means we can ignore the projective issues, we respectively calculate the distances between nose (face center) and mouth/right_eyebrow/left_eyebrow/right_eye/left_eye/jaw and establish a normalized matrix for each face. The classifying based on the distances between any two faces is an LSQ problem. If the distance between sample face and the unknown face is lower than a threshold, then these two faces can be recognized as one person.

- Note: One defect of this algorithm is that it can't resist a photo which has been stretched, which means the face width/height ratio and the distance matrix must be largely different from the same face in the original photo.

## System Requirement
Anaconda3 in ubuntu 18.04

### Env Setup
After Anaconda3 for Pyhon 3.7 is installed (default)
~~~sh
conda create --name env python=3.7
~~~
If python is version 3.6
~~~sh
conda create --name env python=3.6 anaconda
~~~
Then activate
~~~sh
source activate env
~~~
Due to some potential version problems, plz first install tensorflow and related package
~~~sh
conda install tensorflow
conda install matplotlib
pip install opencv-contrib-python
pip install facenet
pip install dlib
pip install keras
~~~

### Main Dependent package
~~~sh
opencv numpy facenet tensorflow scipy matplotlib os dlib keras
~~~

## Running Instruction
Our program is divided into two modes: *all* & *selective*.

### *all* Mode
All detected faces will be covered by a corresponding emoji. For demo here, it will just detect one example image (example1/2/3.jpg), which you can directly change in the *main_all.py file*.
~~~sh
python3 main_all.py
~~~

### *selective* Mode
Only those faces recognized as the same person as the sample face will be covered by emojis. All the parameters users can adjust are list in [*parameters.json*](./parameters.json) for convenience.
~~~sh
python3 main_selective.py
~~~
- Note: If you set the threshold parameter large enough, the *selective* mode will just be the *all* mode, which adds emojis on all detected faces for each photo in album.