# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 17:24
# @Author  : Zongchao Liu
# @FileName: models.py
# @Software: PyCharm

import torch.nn as nn
from .BasicModule import BasicModule


class Conv3d(BasicModule):
    """
    This is a simple 3d CNN with 5 convolutional layers and 3 fully connected layers
    The architecture is as described in:
    Liu Y, Li Z, Ge Q, Lin N and Xiong M (2019) Deep Feature Selection and Causal Analysis of Alzheimer’s Disease. Front. Neurosci. 13:1198. doi: 10.3389/fnins.2019.01198
    """
    def __init__(self, num_classes=2):

        super(Conv3d, self).__init__()

        # kernel size may be changed later!
        self.conv_layer1 = self.make_conv_layer(1, 32, (11, 11, 11), (4, 4, 4), 0)
        self.conv_layer2 = self.make_conv_layer(32, 64, (5, 5, 5), (1, 1, 1), 0)
        self.conv_layer3 = self.make_conv_layer(64, 128, (3, 3, 3), (1, 1, 1), 1)
        self.conv_layer4 = self.make_conv_layer(128, 256, (3, 3, 3), (1, 1, 1), 1)
        self.conv_layer5 = self.make_conv_layer(256, 512, (3, 3, 3), (1, 1, 1), 1)

        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.LeakyReLU()
        self.batch0 = nn.BatchNorm1d(1024, track_running_stats=False)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(1024, track_running_stats=False)
        self.drop = nn.Dropout3d(p=0.15)
        self.fc3 = nn.Linear(1024, num_classes)

    def make_conv_layer(self,in_channel,out_channel, k_size, s, p):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=k_size, padding=p, stride=s),
            nn.LeakyReLU(),
            # pooling layers' parameters may be changed later!
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        #print(x.shape)
        x = self.conv_layer1(x)
        #print("input after conv1: {}".format(x))
        x = self.conv_layer2(x)
        #print("input after conv2: {}".format(x))
        x = self.conv_layer3(x)
        #print("input after conv3: {}".format(x))
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x_1 = x
        x = x.view(x.size(0), -1)
        print("shape before fc: {}".format(x.shape))
        #print("input before fc: {}".format(x))
        x = self.fc1(x)
        x = self.relu(x)
        #print("input before bn0: {}".format(x))
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        #print("input before bn1: {}".format(x))
        x = self.batch1(x)
        x = self.drop(x)
        # x_1 = x
        x = self.fc3(x)

        return x, x_1  # 最后一层，卷积完最后一层


