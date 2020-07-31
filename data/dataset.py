# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 22:37
# @Author  : Zongchao Liu
# @FileName: dataset.py
# @Software: PyCharm

import os
import torchio
import random



random.seed(888)

# Set up the data-set via TorchIO

def make_dataset(imgs, labels):
    subjects = list()
    for (img, label) in zip(imgs, labels):
        print(img, label)
        subject_dict = {
            'FA': torchio.Image(img + '/FA.nii.gz', torchio.LABEL),
            'MD': torchio.Image(img + '/MD.nii.gz', torchio.LABEL),
            'label': label
        }
        print(subject_dict)
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    data_set = torchio.ImagesDataset(subjects)  # remember to add data augmentation options here!
    print('Dataset size:', len(data_set), 'subjects')
    return data_set

# useful visualization functions:

  #def show_sample(sample, image_name, label_name=None):
  #  if label_name is not None:
  #      sample = copy.deepcopy(sample)
  #      #affine = sample[label_name][AFFINE]
  #      label_image = sample[label_name].as_sitk()
  #      label_image = sitk.Cast(label_image, sitk.sitkUInt8)
  #      border = sitk.BinaryContour(label_image)
  #      border_array, _ = torchio.utils.sitk_to_nib(border)
  #      border_tensor = torch.from_numpy(border_array)
  #      image_tensor = sample[image_name][DATA][0]
  #      image_tensor[border_tensor > 0.5] = image_tensor.max()
  #  with tempfile.NamedTemporaryFile(suffix='.nii') as f:
  #      torchio.ImagesDataset.save_sample(sample, {image_name: f.name})
  #      show_nifti(f.name)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    img_path = './data/img_data/'
    labels = ['1', '0']  # would be changed later by the exact formats of labels
    imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]
    make_dataset(imgs, labels)

    # take a look into the sample returned by the dataset
    data_set = make_dataset(imgs, labels)
    sample = data_set[0]
    print(sample)
    fa_map = sample['FA']
    print('Image Keys:', fa_map.keys())

    # set up a data-loader that directly fits torch
    num_subjects = len(data_set)
    training_split_ratio = 0.5
    num_training_subjects = int(training_split_ratio * num_subjects)
    subjects = data_set.subjects


    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(training_subjects)
    validation_set = torchio.ImagesDataset(validation_subjects)

    temp = DataLoader(training_set)
    for _, data in enumerate(temp):
        # tensor already? view(-1, 120, 192, 192, 1) org: torch.Size([2, 1, 192, 192, 120])
        vis_temp = data['FA']['data'][0,0,:,:,:][:, :, 69]
        plt.imshow(vis_temp)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')


