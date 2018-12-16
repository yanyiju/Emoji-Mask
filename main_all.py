# Public packages
import cv2
import matplotlib.pyplot as plt

# Private packages
import face_detection as FACE
import emotion_classification as EMOTION
import emoji_cover as GRAFT

def main():
	# TODO: Choose one example image.
	img_path = "./album/example3.jpg"

	# Face detection process using cv2 package
	# faces, img = FACE.detect_cv2(img_path)
	# FACE.box_cv2(faces, img, img_path)

	# Face detection process using mtcnn package
	# FACE.detect_mtcnn(img_path)

	# Face detection process using dlib package
	gray,faces = FACE.detect_dlib(img_path,'')
	print(faces)
	img = cv2.imread(img_path)

	faces_info = []
	labels = []
	count = 0
	for face in faces:
		face_info = []
		face_info[0:2] = (FACE.get_face_info(gray,face))
		face_path = 'crop_faces/'+str(count)+'.jpg'
		face_img = cv2.imread(face_path)
		# method 0 - CNN
		label = EMOTION.emotion_recognition_CNN(face_img)
		# method 1 - Eigenface
		# label_num = EMOTION.emotion_recognition_EIGEN(face_img)
		# label = EMOTION.get_emotion_name(label_num)
		labels.append(label)
		face_info.append(label)
		count = count+1
		faces_info.append(face_info)
	img_path = 'resize.png'
	result = GRAFT.graft_emoji(img_path,faces_info)
	result_note = FACE.add_box_text(faces,labels,img)
	cv2.imwrite('result.png',result)
	cv2.imwrite('result_note.png',result_note)
	plt.imshow(result)
	plt.show()
	plt.imshow(result_note)
	plt.show()


if(__name__=="__main__"):
    main()