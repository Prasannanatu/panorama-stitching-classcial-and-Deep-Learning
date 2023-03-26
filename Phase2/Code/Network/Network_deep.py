"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(network_output, labels):
    ###############################################
    # Fill your loss function of choice here!
    criterion = nn.MSELoss()
    labels = labels.float()


    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    loss  = criterion(network_output, labels)
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.modelType = modelType
        # if self.modelType == "UnSup":
        #     self.model = unsupervised_net()
        # if self.modelType == "Sup":
        self.model = supervised_Homography_net()

    def forward(self, a,b):
        return self.model(a,b)

    def training_step(self, batch, labels):
        # img_a, patch_a, patch_b, corners, gt  = batch
        delta = self.model(batch)
        loss = LossFn(delta,labels)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(batch, batch_idx)
        loss = LossFn(delta, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class supervised_Homography_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2,64,kernel_size = 3), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,3,1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,3,1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128,256,3,1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,2,1), nn.BatchNorm2d(256), nn.ReLU())
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024,2048)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,8)
        self.flatten = nn.Flatten()
        # self.softmax = nn.softmax /// using regressor model so not necessary
        # self.model = supervised_Homography_net()



    def forward(self,x,labels):
        # print(x.size())
        # print(x)
        # x= torch.concat([xa,xb], axis =2)
        x= self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x = self.maxpool(x)
        x= self.conv4(x)
        x= self.conv5(x)
        x = self.maxpool(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    # def validation_step(self, batch,labels):
    #     # img_a, patch_a, patch_b, corners, gt = batch
    #     delta = supervised_Homography_net(batch,labels)
    #     loss = LossFn(delta, labels)
    #     return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     logs = {"val_loss": avg_loss}
    #     return {"avg_val_loss": avg_loss, "log": logs}




    
        



# class unsupervised_net():

                














































































   