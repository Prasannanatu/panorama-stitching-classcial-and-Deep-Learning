#!/usr/bin/evn python

import numpy as np
import cv2
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import math
import os
import random



def getPatches(image, patch_size = 128, pixel_shift_limit = 32, border_margin = 42):
    
    """
    Inputs: 
    image: image from MS Coco dataset,
    patch_size = size of the patches to be cropped randomly, default: 128
    pixel_shift_limit = radius of pixel neigbourhood for which the corners can be chosen to obtain patch B
    border_margin = margin from the boundaries of image to crop the patch. 
                    (Choose border_margin > pixel_shift_limit)
    
    Returns:
    Patch_a : randomly cropped patch from input image
    Patch_b : patchB cropped from Image B,
    H4: The H4 point homography between Image A and Image B
    pts1,pts2: corner coordinates of Patch_a and Patch_b with respect to Image A and Image B
    """
    
    h,w = image.shape[:2]
    minSize = patch_size+ 2*border_margin+1 # minimuum size of the image to be maintained.
    if ((w > minSize) & (h > minSize)):
        # pixel_shift_limit =  amount of random shift that the corner points can go through
        # make sure border_margin > pixel_shift_limit
         #  leave some margin space along borders 
        end_margin = patch_size + border_margin # to get right/bottom most pixel within image frame
    
    # choose left-top most point within the defined bordera
        x = np.random.randint(border_margin, w-end_margin) # left x pixel of the patch  
        y = np.random.randint(border_margin, h-end_margin) # top y pixel of the patch
    
        # choose left-top most point within the defined border
        pts1 = np.array([[x,y], [x, patch_size+y] , [patch_size+x, y], [patch_size+x, patch_size+y]]) # coordinates of patch P_a
        pts2 = np.zeros_like(pts1)

        # randomly shift coordinates of patch P_a to get coordinates of patch P_b
        for i,pt in enumerate(pts1):
            pts2[i][0] = pt[0] + np.random.randint(-pixel_shift_limit, pixel_shift_limit)
            pts2[i][1] = pt[1] + np.random.randint(-pixel_shift_limit, pixel_shift_limit)

        print(pts1)
        print("pts2", pts2)

        # find H inverse of usin patch coordinates of P_a, P_b
        H_inv = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))) 
        
        imageB = cv2.warpPerspective(image, H_inv, (w,h))

        Patch_a = image[y:y+patch_size, x:x+patch_size]
        Patch_b = imageB[y:y+patch_size, x:x+patch_size]
        H4 = (pts2 - pts1).astype(np.float32) 

        return Patch_a, Patch_b, H4, imageB, np.dstack((pts1,pts2))
    else:
        return None, None, None, None, None
    

def main():
    image = cv2.imread("/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_p1_new/YourDirectoryID_p1/Phase1/Data/Train/Set1/1.jpg")
    output = getPatches(image, patch_size = 128, pixel_shift_limit = 32, border_margin = 42)



if __name__ == '__main__':
    main()