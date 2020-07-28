# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: config.py
# @Software: PyCharm

import os


class DefaultConfig(object):

    # img_path = './data/img_data/'
    labels = ['1', '0']  # would need to be changed later by the exact formats of labels
    imgs = [os.path.join('./data/img_data/', img) for img in sorted(os.listdir('./data/img_data/'))[1:]]

    batch_size = 64  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    img_type = 'FA'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # for loss function

    def parse(self, kwargs):
        """
        update config parameters via an imported dict
        """
        # update settings
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # print --config--
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))