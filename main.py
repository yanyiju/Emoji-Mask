# Public packages
import cv2

# Private packages
import face_detection as FACE
import emoji_cover as GRAFT

def main():
	img_path = "example.jpg"
	# Face detection process using cv2 package
	# faces, img = FACE.detect_cv2(img_path)
	# FACE.box(faces, img, img_path)
	# Face detection process using mtcnn package
	# FACE.detect_mtcnn(img_path)
	gray,faces = FACE.detect_dlib(img_path)
	img = cv2.imread(img_path)

	faces_info = []
	labels = []
	count = 0
	for face in faces:
		face_info = []
		face_info[0:2] = (FACE.get_face_info(gray,face))
		face_path = 'crop_faces/'+str(count)+'.jpg'
		face_img = cv2.imread(face_path)
		label = FACE.emotion_recognition(face_img)
		labels.append(label)
		face_info.append(label)
		count = count+1
		faces_info.append(face_info)
	img_path = 'resize.png'
	result = GRAFT.graft_emoji(img_path,faces_info)
	result_note = FACE.add_box_text(faces,labels,img)
	cv2.imwrite('result.png',result)
	cv2.imwrite('result_not.png',result)


if(__name__=="__main__"):
    main()