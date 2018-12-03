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
	gray, faces = FACE.detect_dlib(img_path)
	print(faces)
	for face in faces:
		print(FACE.get_face_info(gray,face))

if(__name__=="__main__"):
    main()