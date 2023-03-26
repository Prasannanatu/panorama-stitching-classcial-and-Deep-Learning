#!/usr/bin/env python3

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# Add any python libraries here


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    iamge = cv2.imread("/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Train/538.jpg")
    x,y,c = iamge.shape

# 
    # whiteblankimage = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)


    # iamge_1 = cv2.rectangle(iamge, pt1=(65,65), pt2=(193,193), color=(0,0,255), thickness=10)
    # cv2.imwrite("/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/")

    # plt.imshow(whiteblankimage)

    plt.show()
    ts1 = np.array([[65, 65], [193, 65],
                [193, 193], [65, 193],
                ],
               np.float32)
    # iamge2 = cv2.imread("/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/")
    pts = np.array([[113, 2], [91, 219],
                [139, 262], [222, 12],
                ],
               np.float32)


    H = cv2.getPerspectiveTransform(np.float32(ts1), np.float32(pts))
    print(H)
    inverse_homography = np.linalg.inv((H))
    print(inverse_homography)


    warped_image = cv2.warpPerspective(iamge, inverse_homography, (x,y))
    print(warped_image)
    # sys.stdout.buffer.write(cv2.imencode(".jpg", warped_image)[1].tobytes())
    # plt.imshow(warped_image)
    # plt.show()
    
    cv2.imwrite("/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/iamg.jpg",warped_image)
    
    # color = (255, 0, 0)
 
    # Line thickness of 2 px
    # thickness = 2   
    # pts = pts.reshape((-1, 1, 2))


    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
