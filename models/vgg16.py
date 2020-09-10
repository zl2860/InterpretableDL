# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 17:24
# @Author  : Zongchao Liu
# @FileName: models.py
# @Software: PyCharm

import torch.nn as nn


class vgg3d(nn.Module):
    """
    Initially we built a vgg16-like 3d CNN model.

    The architecture is as described in:
    Liu Y, Li Z, Ge Q, Lin N and Xiong M (2019) Deep Feature Selection and Causal Analysis of Alzheimer’s Disease. Front. Neurosci. 13:1198. doi: 10.3389/fnins.2019.01198

    However, such a deep network did not work so well. This does not mean the architecture that we currently is the best.
    You can change the architecture as you want.
    """
    def __init__(self, classify=False):

        super(vgg3d, self).__init__()
        self.classify = classify
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(11, 11, 11), stride=4),
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(5, 5, 5), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            #nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            #nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=1, padding=1),
            #nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            #nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=20)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=20)
        self.gn3 = nn.GroupNorm(num_groups=4, num_channels=20)
        self.gn4 = nn.GroupNorm(num_groups=4, num_channels=20)

        #self.gap = nn.AdaptiveAvgPool3d((None, None, None))

        self.fc = nn.Sequential(
            nn.Linear(in_features=20, out_features=1024),
            #nn.LayerNorm(normalized_shape=4096),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p=0.65),
            nn.LeakyReLU(),
            #nn.Linear(in_features=2048, out_features=2048),
            # nn.LayerNorm(normalized_shape=4096),
            #nn.BatchNorm1d(num_features=2048),
            #nn.Dropout(p=0.75),
            #nn.LeakyReLU()
        )

        self.fc_last = nn.Linear(in_features=1024, out_features=2 if self.classify else 1)
        #nn.init.constant_(self.fc_last.bias, 61)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.gn1(x)
        #print("input after conv1: {}".format(x))
        x = self.conv2(x)
        x = self.gn2(x)
        #print("input after conv2: {}".format(x))
        x = self.conv3(x)
        x = self.gn3(x)
        #print("input after conv3: {}".format(x))
        x = self.conv4(x)
        x = self.gn4(x)
        x = self.conv5(x)
        x = self.gn4(x)
        #x = self.gap(x)
        x_1 = x

        #x = self.gap(x)
        x = x.view(x.shape[0], -1)
        print(x)
        print("input before fc: {}".format(x.shape))
        x = self.fc(x)
        x = self.fc_last(x)

        return x, x_1  # x: last layer output; x_1 layer output before fc


class ADnetwork(nn.Module):
    """
    architecture from the paper shared by Natalie
    """
    def __init__(self, classify=False):
        super(ADnetwork, self).__init__()

        self.classify = classify

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(7, 7, 7), stride=3),
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(3, 3, 3), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=185220, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU()
        )

        self.fc_last = nn.Linear(in_features=128, out_features=3 if self.classify else 1)

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


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    m = ADnetwork(classify=False).cuda()
    rand = torch.rand(size=(10, 1, 190, 190, 190)).cuda()
    output = m(rand)[0]