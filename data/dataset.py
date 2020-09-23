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
def create_train_val(target='score', transform=False, img_type=opt.img_type, weight_sampler=False):
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


def check_fa(dataloader):
    """
    calculate the FA values for each subject
    """
    fa_all = []
    id_all = []
    for _, data in enumerate(dataloader):
        img_batch = data['img']['data']
        print(img_batch.shape)
        id = data['id']
        for idx in range(img_batch.shape[0]):
            img_single = img_batch[idx, :, :, :, :].squeeze()
            fa = torch.mean(img_single)
            fa_all.append(fa)
        id_all.append(id)
    return id_all, fa_all


def check_coverage(dataloader):
    """
    calculate the FA values for each subject
    """
    coverage_all = []
    id_all = []
    for _, data in enumerate(dataloader):
        img_batch = data['img']['data']
        print(img_batch.shape)
        id = data['id']
        for idx in range(img_batch.shape[0]):
            img_single = img_batch[idx, :, :, :, :].squeeze()
            coverage = torch.sum(img_single==0)/(190.0**3)
            coverage_all.append(coverage)
        id_all.append(id)
    return id_all, coverage_all


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visual_check(dataloader, dpi=800):
    df = pd.DataFrame()
    for batch_idx, data in enumerate(dataloader):

        create_dir_not_exist('./visual_check/slice1')
        create_dir_not_exist('./visual_check/slice2')
        create_dir_not_exist('./visual_check/slice3')

        print("-------------Batch{}-------------".format(batch_idx + 1))
        # check shapes
        print(data['img']['data'].shape)
        print(len(data['id']))
        # display slices on dashboard
        grid_1, grid_2, grid_3 = display_slice(data)

        im_show(grid_1)
        plt.savefig('./visual_check/slice1/slice1_batch{}'.format(batch_idx + 1), dpi=dpi)
        #
        im_show(grid_2)
        plt.savefig('./visual_check/slice2/slice2_batch{}'.format(batch_idx + 1), dpi=dpi)
        #
        im_show(grid_3)
        plt.savefig('./visual_check/slice3/slice3_batch{}'.format(batch_idx + 1), dpi=dpi)


        batch_column = pd.Series(data=data['id'], name='batch_{}'.format(batch_idx+1))
        df = df.append(batch_column)
    #df.to_csv('./visual_check/visual_check_list.csv')

    return df


def visual_check_single(id, type='FA'):
    img_path = './data/img/sub-{}/'.format(id)
    id_folder = 'sub-' + id + '_{}.nii.gz'.format(type)
    img = torchio.Image(path=img_path+id_folder, type=torchio.INTENSITY, check_nans=False)
    img = img['data'].squeeze()
    print(img.shape)
    plt.figure('Visualization - {} - {}'.format(type, id))
    plt.subplot(1,3,1)
    plt.imshow(img[95, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, 95, :], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, 95], cmap='gray')
    plt.show()



if __name__ == "__main__":

    # ignore the following

    from torch.utils.data import DataLoader
    import seaborn as sns
    img_type = 'FA'
    train_loader, val_loader = create_train_val(weight_sampler=False)

    #visual check
    visual_check_list = visual_check(train_loader)
    visual_check_list.to_csv('./visual_check/visual_check_list.csv')

    # check low image coverage
    id_coverage, coverage = check_coverage(train_loader)
    coverage = list(torch.tensor(coverage).numpy())
    sns.distplot(coverage)
    # check FA
    id_FA, fa = check_fa(train_loader)
    fa = list(torch.tensor(fa).numpy())
    sns.distplot(fa)

    # only for check input shape
    epoch = 0
    for batch_idx, data in enumerate(train_loader):
        print("-------------Epoch {} Batch{}-------------".format(epoch + 1, batch_idx))
        # check shapes
        print(data['img']['data'].shape)

    print('Training set:', len(train_loader.dataset), 'subjects')
    print('Validation set:', len(val_loader.dataset), 'subjects')


