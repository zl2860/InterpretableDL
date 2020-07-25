# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: main.py
# @Software: PyCharm

import os
import torchio

img_path = './data/img_data/'
labels = ['1', '0']  # would be changed later by the exact formats of labels
imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]