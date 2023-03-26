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
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import tensorflow as tf
import csv
import ast
import re

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")
# Don't generate pyc codes
sys.dont_write_bytecode = True

def ReadLabels_(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        with open(LabelsPathTrain, 'r') as labels_file:
            reader = csv.reader(labels_file)
            labels_iamges = {row[0]: ast.literal_eval(row[1]) for row in reader}
    return labels_iamges

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


def TestOperation(Testset, Coordinates,ModelPath, LabelsPathPred):
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
    model = HomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    OutSaveT = open(LabelsPathPred, "w")

    # for count in tqdm(range(len(TestSet))):
    #     Img, Label = TestSet[count]
    #     Img, ImgOrg = ReadImages(Img)
    print('yoyoyo', Testset.dtype)
    # print('yoyoyo111', Coordinates.dtype)
    print(Coordinates)
    PredT = torch.argmax(model(Testset.to(device),Coordinates.to(device))).item()

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


def ReadLabels( LabelsPathPred):
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

    return  LabelPred


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

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Checkpoints/sup4/11pmodel.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/home/chahatdeep/Downloads/aa/CMSC733HW0/CIFAR10/Test/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    # LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll(BasePath)


    image_2_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Test_/patch_image/Original_warped/"
    image_1_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/Test_/patch_image/Original/"
    patch_a_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels_patch_1.csv"
    patch_b_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels_patch_2.csv"
    labels_path = "/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Data/labels.csv"
    # LabelsPathTrain = os.path.join(labels_path)
    TrainCoordinates = ReadLabels_(labels_path)
    print(TrainCoordinates)
    labels_ = TrainCoordinates['538']
    for key in TrainCoordinates:
        TrainCoordinates[key] = [float(x) for x in TrainCoordinates[key]]
    labels_ = TrainCoordinates['538']
    # labels_ = float(labels_)
    image_1 = cv2.imread(image_1_path + "538.jpg")
    image_2 = cv2.imread(image_2_path + "538.jpg")
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_1 = image_1.reshape(128, 128, 3)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_2 = image_2.reshape(128, 128, 3)
    TrainCoordinpatccates = ReadLabels_(patch_a_path)
    
    for key in TrainCoordinpatccates:
        TrainCoordinpatccates[key] = [float(x) for x in TrainCoordinpatccates[key]]
    patch_a = TrainCoordinpatccates['538']
    TrainCoordinpatccates111 = ReadLabels_(patch_b_path)
    
    for key in TrainCoordinpatccates111:
        TrainCoordinpatccates111[key] = [float(x) for x in TrainCoordinpatccates111[key]]
    patch_b = TrainCoordinpatccates111['538']
    stacked_images = np.concatenate((image_1, image_2), axis=2)
    stacked_image = torch.from_numpy(stacked_images)
    stacked_image = stacked_image.view(1, 6, 128, 128)
    # stacked_image = stacked_image.type(torch.FloatTensor).to(device)
    # stack_iamge =stacked_image = torch.cat(
    #                     [torch.from_numpy(image_1), torch.from_numpy(image_2)], dim=2).to(device)
    image_1 = image_1.astype(np.float32)
    image_1 = torch.from_numpy(image_1).float()
    image_2 = image_2.astype(np.float32)
    image_2 = torch.from_numpy(image_2).float()
    labels_ = np.array(labels_).astype(np.float32)
    labels_ = torch.from_numpy(labels_).float()
    stacked_image = stacked_image.type(torch.FloatTensor).to(device)
    labels_ = labels_.type(torch.FloatTensor).to(device)
    



    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    # LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels
    LabelsPathPred = "./TxtFiles/PredOut.txt"
    # input.float()
    TestOperation(stacked_image, labels_, ModelPath, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsPred = ReadLabels(LabelsPathPred)
    print(LabelsPred)
    # ConfusionMatrix(labels_, LabelsPred)


if __name__ == "__main__":
    main()
