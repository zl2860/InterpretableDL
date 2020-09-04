# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: main.py
# @Software: PyCharm

import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from config import opt
from data.dataset import make_dataset, create_train_val, display_slice, im_show
from torch.utils.data import DataLoader
from torchnet import meter
import torchio
from models.vgg16 import vgg3d
from tqdm import tqdm
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
)  # options for transformation using TorchIO, a tool for loading medical images into torch data-loaders


def train(load=False, target='score', lr=opt.lr, img_type=opt.img_type, transform=False):
    '''
    :param load: Load pre-trained model if True
    :param target: 'score': use the original attention score(scale:50~100) to train;
                   'score_center': use centered score(score-mean_score) to train
    :param lr: learning rate, default:0.001
    :param img_type: map type, default:'FA'
    :param transform: preprocess data using different techniques if True
    :return: None
    '''
    # initial settings
    global_step = 0
    if not load:
        model = vgg3d().cuda()
    else:
        model = torch.load('./checkpoints/vgg3d_score_0901_n.pth').cuda()

    # loss & optimizer
    criterion = torch.nn.MSELoss()  # only tried MAE here, you can customize the loss or use other choices
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=opt.weight_decay) # only tried Adam here, you can use other choices
    # statistical results
    loss_meter = meter.AverageValueMeter()  # for tracking average training loss for every epoch
    pre_loss = 1e10

    # create training and validation set
    training_loader, val_loader = create_train_val(target=target, img_type=img_type, transform=transform)

    # visualization
    log_dir = './logs/exp_{}/'.format(time.strftime('%m-%d'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter('./logs/exp_{}'.format(time.strftime('%m-%d')))
    train_score = torch.tensor([training_loader.dataset.subjects[i]['target'] for i in range(len(training_loader.dataset))])
    writer.add_histogram('Distribution of Training Data', train_score)

    # train
    for epoch in range(opt.max_epoch):

        # reset for each epoch
        loss_meter.reset()
        running_loss = 0.0  # this is the training loss for each iter(batch)

        # each batch
        for batch_idx, data in enumerate(training_loader):
            print("-------------Epoch {} Batch{}-------------".format(epoch+1, batch_idx+1))
            # check shapes
            print(data['img']['data'].shape)

            # reshape data. Desired: size: (batch_size, 1, 190, 190, 190); (batch_size, channel, H, W, depth)
            input_data = data['img']['data'].view(-1, 1, 190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)

            # display slices on dashboard
            grid_1, grid_2, grid_3 = display_slice(data)
            im_show(grid_1)
            writer.add_image('slice_1 of epoch {} batch {}'.format(epoch+1, batch_idx+1), grid_1)
            im_show(grid_2)
            writer.add_image('slice_2 of epoch {} batch {}'.format(epoch+1, batch_idx+1), grid_2)
            im_show(grid_3)
            writer.add_image('slice_3 of epoch {} batch {}'.format(epoch+1, batch_idx+1), grid_3)

            # preprocess input data
            input_data[torch.isnan(input_data)] = 0  # Nan all set to be 0
            input_data[input_data >= 1] = 1.0  # 1 or *over 1 all set to be 1.0

            # get labels
            targets = torch.tensor(list(map(float, data['target'])))   # dtype: int

            # check data and the shapes: targets, input_size, target_size, one_slice_shape
            print("lr: {}".format(lr))
            print("targets : {}".format(targets))
            print("input size: {}".format(input_data.shape))
            print("target size: {}".format(targets.shape))
            print("one slice img size: {}".format(input_data[0, :, :, :, :].view(190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)[(190-opt.crop_size*2)//2, :, :].shape))

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # training
            input_data = input_data.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            res = model(input_data)  # a list, res[0].data: predicted results; res[1]: layer output before fc layers
            print("output size: {}".format(res[0].shape))

            loss = criterion(res[0].squeeze(), targets)
            print("train_loss: {}".format(loss.data))

            loss.backward()
            optimizer.step()
            global_step += 1
            running_loss += loss.item()

            loss_meter.add(loss.data.cpu())
            predicted = res[0].squeeze()
            print("train prediction: {}".format(predicted))

            # validation
            val_loss, val_running_loss = val(model, val_loader)  # test for each iter

            if batch_idx % opt.print_freq == 0:
                writer.add_scalar('training loss',
                                  running_loss,
                                  epoch * len(training_loader) + batch_idx)
                writer.add_scalar('val loss',
                                  val_loss,
                                  epoch * len(training_loader) + batch_idx)

                for i, (name, param) in enumerate(model.named_parameters()):
                    if 'bn' not in name:
                        writer.add_histogram('weights distribution', param, 0)

                running_loss = 0.0
            print("------------------------------------------")
            print(" ")

        # save model for every epoch
        torch.save(model, './checkpoints/exp_{}_{}.pth'.format(name, time.strftime('%m-%d')))

        if loss_meter.value()[0] > pre_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        pre_loss = loss_meter.value()[0]


# validation
def val(model, dataloader):
    """
    validation
    """
    # settings
    criterion = torch.nn.MSELoss()
    model.eval()
    val_step = 0
    val_loss_meter = meter.AverageValueMeter()
    running_loss = 0.0

    for batch_idx, data in tqdm(enumerate(dataloader)):
        print(data['img']['data'].shape)
        val_input = data['img']['data'].view(-1, 1, 190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)
        print("val input size: {}".format(val_input.shape))
        val_label = torch.tensor(list(map(float, data['target'])))

        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()

        with torch.no_grad():
            val_res = model(val_input)

        val_loss = criterion(val_res[0].squeeze(), val_label)  # predicted, label
        running_loss += val_loss
        val_loss_meter.add(val_loss.data.cpu())

        predicted = val_res[0].squeeze()
        print("val prediction: {}".format(predicted))
        print("val diff: {}".format(torch.abs(val_label-predicted)))
        print("val avg loss: {}".format(torch.mean(torch.abs(val_label-predicted))))

        val_step += 1

    model.train()
    return val_loss_meter.value()[0].cpu(), running_loss


if __name__ == "__main__":
    import fire
    fire.Fire()

