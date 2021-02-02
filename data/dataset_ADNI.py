# -*- coding: utf-8 -*-
# @Time    : 1/17/21 2:41 PM
# @Author  : Zongchao Liu
# @FileName: dataset_ADNI.py
# @Software: PyCharm

import os
import torch
from config import opt
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchio as tio
import random
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import torchvision
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    Crop,
    OneOf,
    Compose,
)

# Set up the data-set via TorchIO
def make_dataset():

    img_path = './data/ADNI_T1/'
    subj = os.listdir(img_path)[2:30]
    random.shuffle(subj)
    imgs = [os.path.join(img_path, sub) for sub in subj]
    outcome = pd.read_csv('./data/label.csv')[['Group', 'subID']]

    subj_list = []

    for img in imgs:
        label = outcome.loc[outcome['subID'] == img[15:-7]]['Group'].values
        print('img path: {}\nlabel: {}'.format(img, label))

        subject = tio.Subject({
            'img': tio.Image(img, tio.INTENSITY, check_nans=False),
            'target': label,
            'id': img.split('/')[3][:-7],
        })

        print(subject['img']['data'].shape)

        subj_list.append(subject)

    subjects_dataset = tio.SubjectsDataset(subj_list)#, transform=transform)
    print('Dataset size:', len(subjects_dataset), 'subjects')
    return subjects_dataset


def create_train_val(weight_sampler=False, transform=False):
    data_set = make_dataset()
    training_subjects = [data_set._subjects[i] for i in range(len(data_set._subjects))]
    validation_subjects = [data_set._subjects[i] for i in range(len(data_set._subjects))]

    #training_sampler = WeightedRandomSampler(weights=training_weights,
    #                                         num_samples=len(training_weights),
    #                                         replacement=True)

    # data augmentation or image preprocess
    training_transform = Compose([
        Crop((opt.crop_size, opt.crop_size, opt.crop_size)),
        ZNormalization()
    ])

    val_transform = Compose([
        Crop((opt.crop_size, opt.crop_size, opt.crop_size)),
        ZNormalization()
    ])

    training_set = tio.SubjectsDataset(training_subjects, transform=training_transform if transform else None)
    validation_set = tio.SubjectsDataset(validation_subjects, transform=val_transform if transform else None)

    training_loader = DataLoader(training_set, batch_size=opt.batch_size, drop_last=False,
                                 sampler=training_sampler if weight_sampler else None,
                                 shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=opt.batch_size, drop_last=True)
    return training_loader, val_loader


def display_slice(data):
    vis_1 = (data['img']['data'][:, 0, 95, :, :]).unsqueeze(dim=1)
    vis_2 = (data['img']['data'][:, 0, :, 95, :]).unsqueeze(dim=1)
    vis_3 = (data['img']['data'][:, 0, :, :, 95]).unsqueeze(dim=1)
    grid_1 = torchvision.utils.make_grid(vis_1)
    grid_2 = torchvision.utils.make_grid(vis_2)
    grid_3 = torchvision.utils.make_grid(vis_3)
    return grid_1, grid_2, grid_3


def im_show(grid):
    plt.clf()
    plt.imshow(grid.permute(1, 2, 0))


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    t_loader, v_loader = create_train_val()