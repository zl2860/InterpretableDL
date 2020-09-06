# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: config.py
# @Software: PyCharm

import os
import warnings
import torch


class DefaultConfig(object):
    # img_path = './data/img_data/'
    env = 'default'
    batch_size = 32  # batch size
    test_batch_size = 2   # for test
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 1  # print info every n epoch
    img_type = 'FA'
    result_file = 'result.csv'
    training_split_ratio = 0.8
    load_model_path = None  # None = not loading models

    max_epoch = 300
    lr = 0.03  # initial learning rate
    lr_decay = 0.90  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # for loss function
    crop_size = 0 # 13

    def parse(self, kwargs):
        """
        update config parameters via an imported dict
        """
        # update settings
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        # print --config--
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()