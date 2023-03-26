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
# termcolor, do (pip install termcolor)

import torch
from torchvision.transforms import ToTensor
import torchvision
from Network.Network_deep import LossFn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim import SGD
from Network.Network_deep import HomographyModel, supervised_Homography_net
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import sys
sys.setrecursionlimit(10000)
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *

import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from random import choice
from tqdm import tqdm
# import dircache


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, Process="Train"):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    MiniBatchSize = 8
    # print("iamgen   ", type(ImageNum))
    # print("MiniBatchSize type ", type(MiniBatchSize))

    while ImageNum < MiniBatchSize:

        # print(ImageNum)
        # Generate random image
        # RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        #

        # selected_Image = choice(DirNamesTrain)
        if Process == "Validation":
            print("Validation")

            DirNamesTrain = os.path.join(
                BasePath, "val_/patch_image/Original/")
            selected_Image = choice(DirNamesTrain)

            original_image_path = os.path.join(
                BasePath, "val_/patch_image/original", selected_Image)
            # print(original_image_path)
            patched_image = cv2.imread(original_image_path)

            patched_image = cv2.cvtColor(patched_image, cv2.COLOR_BGR2GRAY)
            original_warped_image_path = os.path.join(
                BasePath, "val_/patch_image/Original_warped", selected_Image)
            warped_image = cv2.imread(original_warped_image_path)
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            selected_Image = selected_Image.rsplit(".", 1)[0]
            # print("will it work",selected_Image)
            h4pt = TrainCoordinates[selected_Image]
            # np.float32(h4pt)

        else:
            LabelsPathTrain = os.path.join('../Data/labels.csv')
            TrainCoordinates = ReadLabels(LabelsPathTrain)
            # print(type(TrainCoordinates))
            dir = '../Data/Train_/patch_image/Original/'
            filename = random.choice(os.listdir(dir))
            path = os.path.join(dir, filename)

            # DirNamesTrain = "../Data/Train_/patch_image/Original/"
            # print(DirNamesTrain)
            # selected_Image = choice(DirNamesTrain)
            # print("This is wrong", filename)
            original_image_path = os.path.join(
                BasePath, "Train_/patch_image/Original/", filename)
            # print("image", original_image_path)
            patched_image = cv2.imread(original_image_path)
            patched_image = cv2.cvtColor(patched_image, cv2.COLOR_BGR2GRAY)
            original_warped_image_path = os.path.join(BasePath, "Train_/patch_image/Original_warped", filename)
            warped_image = cv2.imread(original_warped_image_path)
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            selected_Image = filename.rsplit(".", 1)[0]
            # print("will it work",selected_Image)
            h4pt = TrainCoordinates[selected_Image]
            # np.float32(h4pt)
        # print(patched_image.shape)
        patched_image = patched_image.reshape(128, 128, 1)

        # patched_image = patched_image.T
        # print(patched_image.shape)
        warped_image = warped_image.reshape(128, 128, 1)

        # warped_image = warped_image.T

        # print(warped_image.shape)
        stacked_image = torch.cat(
            [torch.from_numpy(patched_image), torch.from_numpy(warped_image)], dim = 0)
        stacked_image = stacked_image.view(2, 128, 128)
        stacked_image = stacked_image.float()
        ImageNum += 1
        # print(stacked_image.shape)

        ####################################z = ######################
        # Add any standardization or data augmentation here!
        ##########################################################
        # I1 = np.float32(cv2.imread(RandImageName))
        # Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        I1Batch.append(stacked_image)
        CoordinatesBatch.append(torch.tensor(h4pt))
        # print("ImageNum",ImageNum, type(ImageNum))
        # print("MiniBatchSize",MiniBatchSize, type(MiniBatchSize))
    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,

):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel().to(device)
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=0.0005)
    print("Optimizer Information: \n", Optimizer.state_dict)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + "deep.ckpt")
        # Extract only numbers from the name
        StartEpoch = int(
            "".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, 'Training'
            )
            # print(I1Batch.size)

            model.train()

            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch,CoordinatesBatch)
            LossThisBatch = LossFn(
                PredicatedCoordinatesBatch, CoordinatesBatch)
            print("loss for this batch",LossThisBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "deep_model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")
            Validation_set = os.listdir(os.path.join(
                BasePath, "Val_/patch_image/Original"))
            LabelsPathTrain = os.path.join('../Data/validation/labels.csv')
            TrainCoordinates = ReadLabels(LabelsPathTrain)
            model.eval()
            with torch.no_grad():
                # print("asasasa", type(MiniBatchSize))
                validation_batch, validation_labels = GenerateBatch(
                    BasePath, Validation_set, TrainCoordinates, MiniBatchSize, "Validation")

            result = model.validation_step(validation_batch,validation_labels)

            
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model_deep.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data/",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=100,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1.5",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize,
                NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,

    )


if __name__ == "__main__":
    main()
