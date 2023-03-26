#!/usr/bin/env python3




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
# import pry


def create_data(image,patch_size = 128, pixel_shift = 20, border_margin = 32):
    #To create the patch we need to be able to take centre of the image by maintaining border and to enhance the feature possiblity in the patch
    x,y,c = image.shape
    minimum_size = patch_size + pixel_shift + border_margin
    # print(x,y)
    # print(minimum_size)
    if ((x > minimum_size) and (y > minimum_size)):
        # print(x,y)
        end_corner = patch_size + border_margin 
        x_1 = random.randint(border_margin, x-end_corner)
        y_1 = random.randint(border_margin, y-end_corner)
        patch_1 = np.array([[x_1,y_1], [x_1 + patch_size, y_1],[x_1,y_1+patch_size], [x_1+patch_size, y_1+patch_size]])# dont forget double brackets
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

        perturbation_label = ((patch_2) -(patch_1)).astype(np.float32)
        print(type(perturbation_label))

        return image_patch, warped_patch, perturbation_label,patch_1,patch_2
    

def making_sets():
    original_path = "../Data/patch_image/Original"
    patched_name = os.listdir(original_path)


    original_warped_path = "../Data/patch_image/Original_warped"

    warped_name = os.listdir(original_warped_path)
    options = ["Train", "Test", "Validation"]
    original_Train_path = "../Data/Train_/patch_image/Original"
    if not os.path.exists(original_Train_path):
                os.makedirs(original_Train_path, exist_ok=True)

    original_warped_Train_path = "../Data/Train_/patch_image/Original_warped"
    if not os.path.exists(original_warped_Train_path):
                os.makedirs(original_warped_Train_path, exist_ok=True)


    original_Test_path = "../Data/Test_/patch_image/Original"
    if not os.path.exists(original_Test_path):
                os.makedirs(original_Test_path, exist_ok=True)

    original_warped_Test_path = "../Data/Test_/patch_image/Original_warped"
    if not os.path.exists(original_warped_Test_path):
                os.makedirs(original_warped_Test_path, exist_ok=True)

    Trainset, Testset = train_test_split(patched_name, test_size=0.20, random_state=45)

    for patched_name in Trainset:
           warped_name = patched_name
           shutil.move (os.path.join(original_path, patched_name), os.path.join(original_Train_path, patched_name))
           shutil.move (os.path.join(original_warped_path, warped_name), os.path.join(original_warped_Train_path, warped_name))


    for patched_name in Testset:
           warped_name = patched_name
           shutil.move (os.path.join(original_path, patched_name), os.path.join(original_Test_path, patched_name))
           shutil.move (os.path.join(original_warped_path, warped_name), os.path.join(original_warped_Test_path, warped_name))
    





        







def main():
    label_path = "../Data/labels.csv"
    mode = 'w' if not os.path.exists(label_path) else 'a'
    with open(label_path, mode) as labels_file:
        writer = csv.writer(labels_file)


        label_path = "../Data/labels_patch_1.csv"
        mode = 'w' if not os.path.exists(label_path) else 'a'
        with open(label_path, mode) as labels_file:
            writer1 = csv.writer(labels_file)

            label_path = "../Data/labels_patch_2.csv"
            mode = 'w' if not os.path.exists(label_path) else 'a'
            with open(label_path, mode) as labels_file:
                writer2 = csv.writer(labels_file)

    
    
    
            # with open(label_path, 'a') as labels_file:
            #     writer = csv.writer(labels_file)
                

                MSCOCO_DATASET_PATH = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Train/"

                images = os.listdir(MSCOCO_DATASET_PATH)
                # image = cv2.imread("/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_p1_new/YourDirectoryID_p1/Phase1/Data/Train/Set1/1.jpg")


                path_for_images = [MSCOCO_DATASET_PATH + image for image in images]
                options = ["Train", "Test", "Validation"]
                original_path = "../Data/patch_image/Original"

                if not os.path.exists(original_path):
                            os.makedirs(original_path, exist_ok=True)


                original_warped_path = "../Data/patch_image/Original_warped"

                if not os.path.exists(original_warped_path):
                            os.makedirs(original_warped_path, exist_ok=True)


                
                

                for image_path in path_for_images:
                    image = image_path.split('/')[-1].split('.')[0]# the splitting thing got from the CHATGPT. Understood and implemented.


                    input_image = cv2.imread(image_path)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

                    if input_image.shape[0] > 240 and input_image.shape[1] > 240:
                        # print(image)
                        # if (image == '4232'):
                            # print(input_image.shape[0])
                            # print(input_image.shape[1]))
                            # print(input_image)
                        image_patch, warped_patch, perturbation_label,patch_1,patch_2= create_data(input_image,patch_size = 128, pixel_shift = 64, border_margin = 48)
                        # pry()
                        if (warped_patch is None) or (image_patch is None) or(warped_patch.shape != image_patch.shape):
                            continue
                        #     print("its the image")
                        # print(image)
                        #     print(input_image.shape)
                        #     print("ITS A WAPRED PATCH", warped_patch)
                        
                        cv2.imwrite(os.path.join(original_path+ '/' + image+".jpg"),image_patch)
                        cv2.imwrite(os.path.join(original_warped_path+ '/' + image + ".jpg"), warped_patch)
                        
                        writer.writerow([image, list(np.array(perturbation_label).flatten())])

                        
                        writer1.writerow([image, list(np.array(patch_1).flatten())])

                        writer2.writerow([image, list(np.array(patch_2).flatten())])



                making_sets()


        # dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    
    

    
    
    


        # plt.imshow(output)
        # plt.show

    ""
    ""



if __name__ == '__main__':
    main()