# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 17:24
# @Author  : Zongchao Liu
# @FileName: models.py
# @Software: PyCharm

import torch.nn as nn


class AD_design(nn.Module):
    """
    architecture from the paper shared by Natalie
    """
    def __init__(self, classify=False):
        super(AD_design, self).__init__()

        self.classify = classify

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(5, 5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(5, 5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(5, 5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU()
        )

        self.fc_last = nn.Linear(in_features=128, out_features=2 if self.classify else 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_1 = x
        x = x.view(x.shape[0], -1)
        print(x)
        print("input before fc: {}".format(x.shape))
        x = self.fc(x)
        x = self.fc_last(x)

        return x, x_1  # x: last layer output; x_1 layer output before fc

