import graft as gf
import numpy as np
import cv2

def main():
	# Read the example photo in COLOR mode
	photo = cv2.imread('example.png', 1)

    # Set parameters
    faces_info = [[15, 0.1, 1]]
    '''
    face0_orientation = 15	# degree unit
    face0_radius = 0.1
    face0_lable = 1	# eg. set label 1 as happy
    '''

    # Execute grafting
    result = gf.graft_emoji(photo, faces_info)

    # Print out result
    cv2.imwrite("sample_result.png", result)


if(__name__=="__main__"):
    main()