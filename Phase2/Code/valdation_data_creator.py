#!/usr/bin/evn python

import numpy as np
import cv2
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import math
import os
import random
from torch.utils.data import random_split
import csv
from sklearn.model_selection import train_test_split
import shutil
import pry

def create_data_validation(image,patch_size = 128, pixel_shift = 32, border_margin = 32):
    print("hi in create data")
    #To create the patch we need to be able to take centre of the image by maintaining border and to enhance the feature possiblity in the patch
    x,y = image.shape
    minimum_size = patch_size + pixel_shift + border_margin
    # print(x,y)
    # print(minimum_size)
    if ((x > minimum_size) and (y > minimum_size)):
        # print(x,y)
        end_corner = patch_size + border_margin 
        x_1 = random.randint(border_margin, x-end_corner)
        y_1 = random.randint(border_margin, y-end_corner)
        patch_1 = np.array([[x_1,y_1], [x_1 + patch_size, y_1],[x_1,y_1+patch_size], [x_1+patch_size, y_1+patch_size]],np.float32)# dont forget double brackets
        patch_2 = np.zeros(patch_1.shape)
        # pry()
        for i  in range(len(patch_1)):
            # print(points[0])
            # print(points[1])
            patch_2[i][0] = patch_1[i][0] + random.randint(-pixel_shift, pixel_shift)
            patch_2[i][1] = patch_1[i][1] + random.randint(-pixel_shift, pixel_shift)
        # vec = np.vectorize(np.float_)
        # patch_1 = vec(patch_1)
        # patch_2 = vec(patch_2)
        # print(patch_1)  
        # print("patch_2: ", patch_2  )

        H = cv2.getPerspectiveTransform(np.float32(patch_1), np.float32(patch_2))
        inverse_homography = np.linalg.inv((H))


        warped_image = cv2.warpPerspective(image, inverse_homography, (x,y))

        image_patch = image[x_1 : x_1 + patch_size, y_1 : y_1 + patch_size]
        warped_patch = warped_image[x_1 : x_1 + patch_size, y_1 : y_1 + patch_size]
        # if warped_patch.shape != image_patch.shape:
        #     continue
        # cv2.imshow("iamge", warped_image)
        # cv2.waitKey(0)

        perturbation_label = patch_2 - patch_1
        # pry()

        return image_patch, warped_patch, perturbation_label






def making_sets():
    print("hi in making_set")
    original_path = "../Data/validation/patch_image/Original"
    patched_name = os.listdir(original_path)


    original_warped_path = "../Data/validation/patch_image/Original_warped"

    warped_name = os.listdir(original_warped_path)
    options = ["Train", "Test", "Validation"]
    original_Train_path = "../Data/Val_/patch_image/Original"
    if not os.path.exists(original_Train_path):
                # pry()
                os.makedirs(original_Train_path, exist_ok=True)

    original_warped_Train_path = "../Data/Val_/patch_image/Original_warped"
    if not os.path.exists(original_warped_Train_path):
                os.makedirs(original_warped_Train_path, exist_ok=True)


    original_Test_path = "../Data/Val_/patch_image/Original"
    if not os.path.exists(original_Test_path):
                os.makedirs(original_Test_path, exist_ok=True)

    original_warped_Test_path = "../Data/Val_/patch_image/Original_warped"
    if not os.path.exists(original_warped_Test_path):
                os.makedirs(original_warped_Test_path, exist_ok=True)
    validation_Set = os.listdir()

    for patched_name in Validation_set:
           warped_name = patched_name
           shutil.move (os.path.join(original_path, patched_name), os.path.join(original_Train_path, patched_name))
           shutil.move (os.path.join(original_warped_path, warped_name), os.path.join(original_warped_Train_path, warped_name))



    


    

def main():
    # print("hi in main")
    if not os.path.exists("../Data/validation/"):
        os.makedirs("../Data/validation/")
    # pry()
    label_path = "../Data/validation/labels.csv"
    mode = 'w' if not os.path.exists(label_path) else 'a'
    with open(label_path, mode) as labels_file:
        writer = csv.writer(labels_file)


    
    
    # with open(label_path, 'a') as labels_file:
    #     writer = csv.writer(labels_file)

        MSCOCO_DATASET_PATH = "/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_p1_new/YourDirectoryID_p1/Phase2/Data/Val/"
        # pry()
        images = os.listdir(MSCOCO_DATASET_PATH)    
        # pry()
        # image = cv2.imread("/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_p1_new/YourDirectoryID_p1/Phase1/Data/Train/Set1/1.jpg")


        path_for_images = [MSCOCO_DATASET_PATH + image for image in images]
        # options = ["Train", "Test", "Validation"]
        original_path = "../Data/validation/patch_image/Original"

        if not os.path.exists(original_path):
                    os.makedirs(original_path, exist_ok=True)


        original_warped_path = "../Data/validation/patch_image/Original_warped"

        if not os.path.exists(original_warped_path):
                    os.makedirs(original_warped_path, exist_ok=True)


        
        

        for image_path in path_for_images:
            print("hi")
            image = image_path.split('/')[-1].split('.')[0]# the splitting thing got from the CHATGPT. Understood and implemented.


            input_image = cv2.imread(image_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            if input_image.shape[0] > 240 and input_image.shape[1] > 240:
                image_patch, warped_patch, perturbation_label= create_data_validation(input_image,patch_size = 128, pixel_shift = 64, border_margin = 48)
                if (warped_patch is None) or (image_patch is None) or(warped_patch.shape != image_patch.shape):
                    continue
                cv2.imwrite(os.path.join(original_path+ '/' + image+".jpg"),image_patch)
                cv2.imwrite(os.path.join(original_warped_path+ '/' + image + ".jpg"), warped_patch)
                writer.writerow([image, list(np.array(perturbation_label).flatten())])


        making_sets()






    









if __name__ == '__main__':
    main()