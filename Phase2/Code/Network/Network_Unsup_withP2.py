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
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(network_output, labels):
    ###############################################
    # Fill your loss function of choice here!
    # criterion = nn.MSELoss()
    # labels = labels.float()
    # print(network_output)
    # loss  = torch.mean(torch.sum(abs(network_output - labels)))
    labels = labels.unsqueeze(1)
    criterion = nn.L1Loss()
    loss = F.l1_loss(labels, network_output, reduction='mean')
    print (loss)

    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    # loss  = criterion(network_output, labels)
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.modelType = modelType
        # if self.modelType == "UnSup":
        #     self.model = unsupervised_net()
        # if self.modelType == "Sup":
        self.model = Net()
    def forward(self,a,b,c,d,e,f):
        return self.model(a,b,c,d,e,f)

    def training_step(self, batch, labels):
        # img_a, patch_a, patch_b, corners, gt  = batch
        delta = self.model(batch)
        loss = LossFn(delta,labels)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx,c,d,e,f):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(batch, batch_idx,c,d,e,f)
        loss = LossFn(delta, f)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}





class Net(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        ...
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.conv1 = nn.Sequential(nn.Conv2d(2,12,kernel_size = 3), nn.BatchNorm2d(12), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(12,36,3,1), nn.BatchNorm2d(36), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(36,36,3,1), nn.BatchNorm2d(36), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(36,48,3,1), nn.BatchNorm2d(48), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(48,48,3,1), nn.BatchNorm2d(48), nn.ReLU())
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3072,1024)
        self.fc2 = nn.Linear(1024,8)
        self.flatten = nn.Flatten()



        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    


        


        



    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x, pb):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, xb,p_a,p_b, image_1,image_2):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!

        x= self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x = self.maxpool(x)
        x= self.conv5(x)
        x= self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = Tensor_DLT(x, p_a)

        # print("please display",image_1.shape)
        # print("please display2 ",x.shape)
        x = torch.Tensor(x)
        # x.requires_grad = True
        image_1 = image_1.reshape(32,1,128,128)
        patch_b_pred = kornia.geometry.transform.warp_perspective(image_1, x, dsize = (128,128),
                                                                     mode='bilinear', padding_mode='zeros', 
                                                                     align_corners=True, fill_value=torch.zeros(3))
        print(patch_b_pred.shape)
        return patch_b_pred

        #############################
        
def Tensor_DLT(H4pt, C4pt_A):
    import numpy as np
    # print("yey ",C4pt_A.size())
    C4pt_B = H4pt + C4pt_A
    A_ = np.empty((8, 8))
    # b_ = np.empty((2, 1))
    H_ = np.empty((8,1))
    H_all = np.empty((32,3,3))
    b_ = np.empty((8,1))

        # C4pt_B = H4pt + C4pt_A
    values = [0, 2, 4, 6]
        # for val in vals:
    # print("oho", C4pt_A.shape[0])
    for i in range(C4pt_A.shape[0]):

        for val in values:
            # print(i)
            u_i = C4pt_A[i,val]
            v_i = C4pt_A[i,val+1]
            u_pi = C4pt_B[i,val]
            v_pi = C4pt_B[i,val+1]
            A = torch.tensor([0, 0, 0, -u_i, -v_i, -1, (-u_i)*v_pi, (v_i*v_pi)])
            A = A.numpy()
            B = torch.tensor([u_i, v_i, 1, 0, 0, 0, -u_pi*u_i, -u_pi*v_i])
            B = B.numpy()


            # A_[0,:] = np.vstack((A_, A)) if A_.size else A
            # print("in between", A_.shape)
            # A_[0:1] = np.vstack((A_, B)) if A_.size else A

            A_[val,:] = A
            A_[val+1,:] = B
            b = torch.tensor([-v_pi,u_pi])
            b = b.numpy()
            b_[val:val+2,0] = b.T
        # b.reshape(2,1)
        # print("a_.shape",A_.shape)
        # print("b_.shape",b_.shape)
        A_inv = np.linalg.pinv(A_)
        H = np.dot(A_inv,b_)
        H_ = np.append(H, [[1]], axis=0)
        H_ = np.double(H_)
        H_ = H_.reshape(1,3,3)
        H_all[i,:,:] = H_
    H_all = torch.from_numpy(H_all)
    H_all.requires_grad = True    





        
        # H_ = np.vstack((H_, H)) if b_.size else b

    # print("me aya",H_all.shape)

    # print(H_all)
    # print("space") 

    return H_all     
