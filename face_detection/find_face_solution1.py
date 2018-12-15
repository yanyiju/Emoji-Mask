import cv2
import matplotlib.pyplot as plt

def detect(path):
    img = cv2.imread(path)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img, path):
    idx = 0
    for x1, y1, x2, y2 in rects:
        sub_img = img[y1 : y2, x1 : x2]
        cv2.imwrite('detected_faces/' + path + '_' + str(idx) + '.jpg', sub_img)
        cv2.imwrite('detected_faces/' + path + '_' + str(idx) + '.jpg', sub_img)
        idx = idx + 1
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite('detected_cluster.jpg', img)

img_name = "example2.jpg"
rects, img = detect(img_name)
box(rects, img, img_name)
