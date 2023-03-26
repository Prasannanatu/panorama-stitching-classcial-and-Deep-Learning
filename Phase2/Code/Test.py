#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from random import choice
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network_Unsup import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import tensorflow as tf
import csv
import ast

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
def ReadLabels_(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
            print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
            sys.exit()
    else:
        with open(LabelsPathTrain, 'r') as labels_file:
            reader = csv.reader(labels_file)
            labels_iamges = {row[0]: ast.literal_eval(row[1]) for row in reader}
    return labels_iamges


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def TestOperation(TestSet,labels,patch_a,patch_b,iamge_1, image_2,ModelPath,LabelsPathPred):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    LabelsPathTrain = os.path.join('../Data/labels.csv')
    # patch_a_ = ReadLabels_(patch_a)
    # patch_b_ = ReadLabels_(patch_b)
    # iamge_1_ = ReadLabels_(iamge_1)
    # image_2_path = ReadLabels_(image_2)


    path = "path/to/folder/"
    number = 123
    img_path = path + str(number) + ".jpg"

    # Read the image using OpenCV
    
    img = cv2.imread(img_path)
    
    model = HomographyModel()

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    OutSaveT = open(LabelsPathPred, "w")

    for count in tqdm(range(len(TestSet))):
        # Img = TestSet[count]
        # patch_a = patch_a_[count]
        # patch_b = patch_b_[count]
        # image_1_path = iamge_1 + str(count) + ".jpg"
        # image_1_  = cv2.imread(image_1_path)
        # image_1_  = cv2.cvtColor(image_1_, cv2.COLOR_BGR2GRAY)

        # image_2_path = iamge_2 + str(count) + ".jpg"
        # image_2_  = cv2.imread(image_2_path)
        # image_2_  = cv2.cvtColor(image_2_, cv2.COLOR_BGR2GRAY)

        # labels =  ReadLabels_(LabelsPathTrain)
        # Img, ImgOrg = ReadImages(Img)
        PredT = torch.argmax(model(TestSet,labels,patch_a,patch_b,iamge_1, image_2)).item()
        print("yey",PredT)

        OutSaveT.write(str(PredT) + "\n")
    OutSaveT.close()


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels (LabelsPathPred):
    # if not (os.path.isfile(LabelsPathTest)):
    #     print("ERROR: Test Labels do not exist in " + LabelsPathTest)
    #     sys.exit()
    # else:
    #     LabelTest = open(LabelsPathTest, "r")
    #     LabelTest = LabelTest.read()
    #     LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """
    loss_T = []
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Checkpoints/unsup4/9a0unsup4model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Val/",
        help="Path to load images from, Default:BasePath",
    )
    # Parser.add_argument(
    #     "--LabelsPath",
    #     dest="LabelsPath",
    #     default="./TxtFiles/LabelsTest.txt",
    #     help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    # LabelsPath = Args.LabelsPath

    image_2_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Test_/patch_image/Original_warped/"
    image_1_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Test_/patch_image/Original/"
    patch_a_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels_patch_1.csv"
    patch_b_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels_patch_2.csv"
    labels_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels.csv"
    I1Batch = []
    labels =[]
    image_11= []
    image_22 =[]
    patch_aa = []
    patch_bb =[]
    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll()

    # iamge_11 = os.listdir(image_1_path)
    # num_images = len([f for f in iamge_11 if f.endswith('.jpg')])
    for i in range(500):
        # selected_Image_ = choice(os.listdir(image_1_path))
        # print(selected_Image_)
        # selected_Image = selected_Image_.rsplit(".", 1)[0]
        # print(selected_Image)
        
        norm = np.zeros((128,128))
        image_1_new = os.path.join(image_1_path, str(i) + ".jpg")
        if os.path.exists(image_1_new):
            image_1 = cv2.imread(image_1_new)
            if image_1 is None:
                continue
                if selected_Image in labels_path:
                    labels_ = labels_path[selected_Image]
                    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
                    image_1 = cv2.normalize(image_1, norm, 0, 1, cv2.NORM_MINMAX)
                    labels_ = labels_path[i]
                    patch_a = patch_a_path[i]
                    patch_b = patch_b_path[i]

                    image_2_new = os.path.join(image_1_path, str(i) + ".jpg")
                    image_2 = cv2.imread(image_2_new)
                    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
                    image_2 = cv2.normalize(image_2, norm, 0, 1, cv2.NORM_MINMAX)
                    stacked_image = torch.cat(
                        [torch.from_numpy(image_1), torch.from_numpy(image_2)], axis=0)

                    image_11.append(torch.tensor(image_1,dtype=torch.double))
                    image_22.append(torch.tensor(image_2,dtype=torch.double))
                    patch_aa.append(torch.tensor(patch_a,dtype=torch.double))
                    patch_bb.append(torch.tensor(patch_b,dtype=torch.double))
                    stacked_image = stacked_image.view(2, 128, 128)
                    stacked_image = stacked_image.float()
                    I1Batch.append(stacked_image)
                    labels.append(labels_)
        else:
            continue






    
    

    # images_tensors = [torch.tensor(img).permute(1, 128, 128) for img in images]

    
    # images = [Image.open(os.path.join(path, f)) for f in image_files] 
    # images_tensors = [torch.tensor(img).permute(1, 128, 128) for img in images]
    # # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels
    

    TestOperation(I1Batch,labels,patch_aa,patch_bb,image_11, image_22,ModelPath,LabelsPathPred)

    # Plot Confusion Matrix
    # image_22 = list(image_22)
    LabelsPred = ReadLabels(LabelsPathPred)
    print("yesyes just there",type(LabelsPred))
    LabelsPred =list(LabelsPred)

    # image_22 = torch.tensor(image_22)
    image_22 = np.array(image_22)
    LabelsPred = torch.tensor(LabelsPred)
    Labelspred = np.array(LabelsPred)
    print("uypuptjerif", image_22)
    # LabelsPathPred = "./TxtFiles/PredOut.txt"/ 
    # OutSaveT = open(LabelsPathPred, "r")
    ConfusionMatrix(image_22, LabelsPred)
    loss = np.mean(abs(image_22 - LabelsPred))
    loss_T.append(loss)
    print(loss)




if __name__ == "__main__":
    main()
