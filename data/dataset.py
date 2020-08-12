# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 22:37
# @Author  : Zongchao Liu
# @FileName: dataset.py
# @Software: PyCharm

import os
import torchio
import random
import pandas as pd



random.seed(888)

# Set up the data-set via TorchIO
def make_dataset(img_type, target = 'score'):  # target can either be att_score or label(ADHD)
    img_path = './data/img/'
    subj = sorted(os.listdir(img_path))
    imgs = [os.path.join(img_path, sub) + '/{}_{}.nii.gz'.format(sub, img_type) for sub in subj]
    labels = pd.read_csv('./data/label/label.csv').iloc[:, 1:].sort_values('id').iloc[:, 1]
    score = pd.read_csv('./data/score/att_score.csv').sort_values('ID').iloc[:,1]
    subjects = list()
    if target == 'label':
        t = labels


    elif target == 'score':
        t = score

    for (img, t) in zip(imgs, t):
        print('img path: {}\nlabel: {}'.format(img, t))
        subject_dict = {
            'img': torchio.Image(img, torchio.LABEL, check_nans=False),
            'target': t
        }
        print(subject_dict)
        if not pd.isnull(subject_dict['target']):
            subject = torchio.Subject(subject_dict)
            subjects.append(subject)

    data_set = torchio.ImagesDataset(subjects)  # remember to add data augmentation options here!
    print('Dataset size:', len(data_set), 'subjects')
    return data_set

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    img_type = 'FA'
    data_set = make_dataset(img_type, target='score')

    # take a look into the sample returned by the dataset
    # default image has missing values
    sample = data_set[0]
    print(sample)
    fa_map = sample['img']
    print('Image Keys:', fa_map.keys())

    # set up a data-loader that directly fits torch
    num_subjects = len(data_set)
    training_split_ratio = 0.8
    num_training_subjects = int(training_split_ratio * num_subjects)
    subjects = data_set.subjects

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(training_subjects)
    validation_set = torchio.ImagesDataset(validation_subjects)

    temp = DataLoader(training_set)
    for _, data in enumerate(temp):
        # size: (1, 1, 190 ,190 ,190)
        vis_temp = data['img']['data'][0, 0, :, :, :][:, :, 95]
        #plt.imshow(vis_temp)
        print(torch.tensor(list(map(float, data['label'])), dtype=torch.long))

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')


