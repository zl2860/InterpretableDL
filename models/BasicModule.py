# -*- coding: utf-8 -*-
# @Time    : 2020-08-01 14:42
# @Author  : Zongchao Liu
# @FileName: BasicModule.py
# @Software: PyCharm


import torch.nn as nn
import torch
import time

class BasicModule(nn.Module):
    """
    for saving and loading models
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        load model from a specific directory
        """
        self.load_state_dict(torch.load(path))

    def save(self, name = None):
        """
        save the model. The default filename is xxxx_mmdd_time.pth
        """

        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S:.pth')

        torch.save(self.state_dict(), name)
        return name
    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)