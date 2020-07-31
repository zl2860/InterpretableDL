# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 17:24
# @Author  : Zongchao Liu
# @FileName: models.py
# @Software: PyCharm

import torch.nn as nn



class Conv3d(nn.Module):
    """
    This is a simple 3d CNN with 5 convolutional layers and 3 fully connected layers
    The architecture is as described in:
    Liu Y, Li Z, Ge Q, Lin N and Xiong M (2019) Deep Feature Selection and Causal Analysis of Alzheimer’s Disease. Front. Neurosci. 13:1198. doi: 10.3389/fnins.2019.01198
    """
    def __init__(self, num_classes):

        super(Conv3d, self).__init__()

        self.conv_layer1 = self.make_conv_layer(1, 128, (4, 4, 4), (4, 4, 4))
        self.conv_layer2 = self.make_conv_layer(128, 256, (2, 2, 1), (1, 1, 1))
        self.conv_layer3 = self.make_conv_layer(256, 512, (2, 2, 1), (1, 1, 1))
        self.conv_layer4 = self.make_conv_layer(512, 1024, (2, 2, 1), (1, 1, 1))
        self.conv_layer5 = self.make_conv_layer(1024, 1024, (2, 2, 1), (1, 1, 1))

        self.fc1 = nn.Linear(491520, 512)
        self.relu = nn.LeakyReLU()
        self.batch0 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout3d(p=0.15)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout3d(p=0.15)
        self.fc3 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def make_conv_layer(self,in_channel,out_channel, k_size, s):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=k_size, padding=0, stride=s),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 1))
        )
        return conv_layer

    def forward(self, x):
        print(x.shape)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = x.view(x.size(0), -1)
        #print(x.size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x_1 = x
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x, x_1  # 最后一层，倒数第二层
