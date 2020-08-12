from .BasicModule import BasicModule
import torch.nn as nn


class SiameseNet(BasicModule):
    '''
    A siamese network for detecting  similarity
    the architecture temporarily shares the same pattern with the previously built simple 3d CNN
    '''
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(4, 4, 4), stride=(4, 4, 4)),
            nn.LeakyReLu(),
            nn.MaxPool3d(2, 2, 2),
            nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.LeakyReLu(),
            nn.MaxPool3d(2, 2, 2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.LeakyReLu(),
            nn.MaxPool3d(2, 2, 2),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 2, 1), stride=(1, 1, 1)),
            nn.LeakyReLu()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024, track_running_stats=False),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.Dropout3d(p=0.15),
            nn.Linear(512, 1)
        )

    def forward_once(self,x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input_left, input_right):
        output_left = self.forward_once(input_left)
        output_right = self.forward_once(input_right)
        return output_left, output_right