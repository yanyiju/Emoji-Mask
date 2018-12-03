import graft as gf
import numpy as np
import cv2

# from PIL import image

def main():
	# Set parameters
	faces_info = []
	faces_info.append([15,100,(500,500),1])
	'''
	face0_orientation = 15	# degree unit
	face0_radius = 0.1
	face0_center = (100, 100)
	face0_lable = 1	# eg. set label 1 as happy
	'''

    # Execute grafting
	photo_path = 'example.png'
	result = gf.graft_emoji(photo_path,faces_info)

	# Print out result
	cv2.imwrite("sample_result.png",result)


if(__name__=="__main__"):
    main()