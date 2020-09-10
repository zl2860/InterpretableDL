# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 22:37
# @Author  : Zongchao Liu
# @FileName: dataset.py
# @Software: PyCharm

import os
import torch
from config import opt
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchio
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

random.seed(888)


# check img shape. There may exist images whose shape is not (190, 190, 190)
def check_tensor_dim(tensor):
    if list(tensor.shape) == [1, 190, 190, 190]:
        return True  # correct
    else:
        return False


# check if the target is missing
def  check_target_value(target):
    if pd.isnull(target):
        return False
    else:
        return True  # correct


# Set up the data-set via TorchIO
def make_dataset(img_type, target='score'):

    '''
    :param img_type: FA, MD etc. Default is FA
    :param target: 'score': original CBCL_attention_score; 'score_center': centered score = score - mean(score)
    :return: TorchIO.dataset object
    '''

    img_path = './data/img/'
    subj = sorted(os.listdir(img_path))
    imgs = [os.path.join(img_path, sub) + '/{}_{}.nii.gz'.format(sub, img_type) for sub in subj]
    score = pd.read_csv('./data/score/att_score_all.csv').sort_values('ID').iloc[:, 1]
    score_group = pd.read_csv('./data/score/att_score_all.csv').sort_values('ID').iloc[:, 2]

    score_center = score - 60.78
    assignment = pd.read_csv('./data/score/att_score_all.csv').sort_values('ID').iloc[:, 4]
    subjects = list()

    if target == 'score_center':
        t = score_center
    elif target == 'score':
        t = score
    else:
        t = score_group

    for (img, outcome, train_val) in zip(imgs, t, assignment):
        print('img path: {}\nlabel: {}'.format(img, outcome))
        subject_dict = {
            'img': torchio.Image(img, torchio.INTENSITY, check_nans=False),
            'target': outcome,
            'id': img.split('/')[3],
            'assignment': train_val
        }
        print(subject_dict['img']['data'].shape)

        # preprocess input data
        if check_target_value(subject_dict['target']) and check_tensor_dim(subject_dict['img']['data']):
            subject_dict['img']['data'][subject_dict['img']['data'] >= 1.0] = 1.0
            subject_dict['img']['data'][torch.isnan(subject_dict['img']['data'])] = 0
            subject = torchio.Subject(subject_dict)
            subjects.append(subject)

    data_set = torchio.SubjectsDataset(subjects)
    print('Dataset size:', len(data_set), 'subjects')
    return data_set


# get training and validation set
def create_train_val(target='score', transform=False, img_type=opt.img_type):
    data_set = make_dataset(img_type=img_type, target=target)
    training_subjects = [data_set.subjects[i] for i in range(len(data_set.subjects)) if data_set.subjects[i]['assignment'] == 'train']
    validation_subjects = [data_set.subjects[i] for i in range(len(data_set.subjects)) if data_set.subjects[i]['assignment'] == 'val']

    # add a weighted sampler, though it is not used in training
    if target == 'score':
        training_weights = [2 if training_subjects[i]['target'] >= 70 else 1 for i in range(len(training_subjects))]
    elif target == 'score_group':
        training_weights = [3 if training_subjects[i]['target'] == 1 else 1 for i in range(len(training_subjects))]

    training_sampler = WeightedRandomSampler(weights=training_weights,
                                             num_samples=len(training_weights),
                                             replacement=True)

    # data augmentation or image preprocess
    training_transform = Compose([
        Crop((opt.crop_size, opt.crop_size, opt.crop_size)),
        ZNormalization()
    ])

    val_transform = Compose([
        Crop((opt.crop_size, opt.crop_size, opt.crop_size)),
        ZNormalization()
    ])

    training_set = torchio.SubjectsDataset(training_subjects, transform=training_transform if transform else None)
    validation_set = torchio.SubjectsDataset(validation_subjects, transform=val_transform if transform else None)

    training_loader = DataLoader(training_set, batch_size=opt.batch_size, drop_last=True, sampler=training_sampler,
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
    plt.imshow(grid.permute(1, 2, 0))


if __name__ == "__main__":

    # ignore the following

    from torch.utils.data import DataLoader
    img_type = 'FA'
    train_loader, val_loader = create_train_val()

    epoch = 0
    for batch_idx, data in enumerate(train_loader):
        print("-------------Epoch {} Batch{}-------------".format(epoch + 1, batch_idx))
        # check shapes
        print(data['img']['data'].shape)

    print('Training set:', len(train_loader.dataset), 'subjects')
    print('Validation set:', len(val_loader.dataset), 'subjects')


