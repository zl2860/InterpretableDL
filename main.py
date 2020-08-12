# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: main.py
# @Software: PyCharm


import torch
from config import opt
from data.dataset import make_dataset
from torch.utils.data import DataLoader
from models.models import Conv3d
from torchnet import meter
import torchio
from torch.utils.data import WeightedRandomSampler
from torch.autograd import Variable
import numpy as np
from visdom import Visdom
from utils.visualize import Visualizer
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
)


#img_path = './data/img_data/'
#labels = ['1', '0']  # would be changed later by the exact formats of labels
#imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]


def train(load=True):
    # model
    #opt = DefaultConfig()
    global_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not load:
        model = Conv3d(num_classes=1)
    else:
        model = torch.load('./checkpoints/conv3d_score.pth')

    # loss & optimizer
    criterion = torch.nn.L1Loss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=opt.weight_decay)
    # statistical results
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    pre_loss = 1e10

    # transformation
    training_transform = Compose([
        #RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        #HistogramStandardization({MRI: landmarks}),
        RandomBiasField(),
        #ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        #ToCanonical(),
        #Resample((4, 4, 4)),
        #CropOrPad((48, 60, 48)),
        RandomFlip(axes=(0,)),
        Crop((opt.crop_size, opt.crop_size, opt.crop_size)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    val_transform = Compose([
        Crop((opt.crop_size, opt.crop_size, opt.crop_size))
    ])

    # create training and validation set
    data_set = make_dataset(opt.img_type)
    num_subjects = len(data_set)
    training_split_ratio = opt.training_split_ratio
    num_training_subjects = int(training_split_ratio * num_subjects)

    training_subjects = data_set.subjects[:num_training_subjects]
    validation_subjects = data_set.subjects[num_training_subjects:]

    #training_weights = [9 if training_subjects[i]['label'] == 0 else 1 for i in range(len(training_subjects))]
    #training_sampler = WeightedRandomSampler(#weights=training_weights,
    #                                         num_samples=len(training_weights),
    #                                         replacement=True)

    training_set = torchio.ImagesDataset(training_subjects, transform=training_transform)
    validation_set = torchio.ImagesDataset(validation_subjects, transform=val_transform)

    training_loader = DataLoader(training_set, batch_size=opt.batch_size, drop_last=True, #sampler=training_sampler,
                                 shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=opt.batch_size, drop_last=True)

    # visualization
    vis = Visdom()
    visualizer = Visualizer(opt.env)

    # train
    for epoch in range(opt.max_epoch):

        # reset for each epoch
        loss_meter.reset()
        confusion_matrix.reset()

        # start training
        for batch_idx, data in enumerate(training_loader):

            print("-------------Epoch {} Batch{}-------------".format(epoch+1, batch_idx))
            # check shapes
            print(data['img']['data'].shape)

            # org: torch.Size([2, 1, 192, 192, 120]), we want depth = 190, channel = 1
            input_data = data['img']['data'].view(-1, 1, 190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)

            # check one slice
            vis_data_1 = (data['img']['data'][0, 0, :, :, :][opt.crop_size:190-opt.crop_size, opt.crop_size:190-opt.crop_size, (190-2*opt.crop_size)//2]) * 255.0
            vis_data_2 = (data['img']['data'][0, 0, :, :, :][(190-2*opt.crop_size)//2, opt.crop_size:190-opt.crop_size, opt.crop_size:190-opt.crop_size]) * 255.0
            vis_data_3 = (data['img']['data'][0, 0, :, :, :][opt.crop_size:190-opt.crop_size, (190-2*opt.crop_size)//2, opt.crop_size:190-opt.crop_size]) * 255.0

            # deal with missing values
            input_data[torch.isnan(input_data)] = 0  # Nan all set to 0

            # get labels
            targets = torch.tensor(list(map(float, data['target'])))  # dtype=torch.long

            # check data and the shapes: targets, input_size, target_size, one_slice_shape
            print("targets : {}".format(targets))
            print("input size: {}".format(input_data.shape))
            print("target size: {}".format(targets.shape))
            print("one slice img size: {}".format(input_data[0, :, :, :, :].view(190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)[(190-opt.crop_size*2)//2, :, :].shape))

            # check the images from one slice
            visualizer.img("slice {}".format("95_1"), vis_data_1)
            visualizer.img("slice {}".format("95_2"), vis_data_2.T)
            visualizer.img("slice {}".format("95_3"), vis_data_3)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # training
            if opt.use_gpu:
                input_data = input_data.to(device)
                targets = targets.to(device)
                model = model.to(device)

            optimizer.zero_grad()
            res = model(input_data)  # a list, res[0].data: prob from output layer; res[1]: x_1
            print("output size: {}".format(res[0].shape))
            #print("output: {}".format(res[0]))
            #print("last layer prob: {}".format(res[0].data))

            #y = torch.zeros(opt.batch_size, 2)
            #y[range(y.shape[0]), targets] = 1
            #print("one hot label: {}".format(y))

            #if opt.use_gpu:
                #y = y.to(device)

            loss = criterion(res[0].squeeze(), targets)
            print("train_loss: {}".format(loss.data))

            loss.backward()
            optimizer.step()
            global_step += 1

            vis.line([loss.item()], [global_step], win='train_loss', update='append')

            loss_meter.add(loss.data.cpu())
            predicted = res[0].squeeze()
            print("train prediction: {}".format(predicted))
            #confusion_matrix.add(predicted, targets.data)

            # validation
            val_loss = val(model, val_loader)  # it is a meter obj

            visualizer.plot('val_loss', val_loss)

            #print(confusion_matrix)

            if batch_idx % opt.print_freq == 0:
                visualizer.plot('train_loss', loss_meter.value()[0].cpu())  # loss_meter.value()[0]: mean value of loss
            print("------------------------------------------")
            print(" ")

        if epoch % opt.print_freq == 0:
            print("Epoch: {}/{} | loss: {}".format(epoch, opt.max_epoch, loss_meter.value()[0].cpu()))

        #model.save()
        torch.save(model, './checkpoints/conv3d_score.pth')

        visualizer.log("epoch:{epoch}    lr: {lr}    train_loss: {train_loss} val_loss: {val_loss}".format( #train_cm:{train_cm}    val_cm:{val_cm}
            epoch=epoch,
            lr=opt.lr,
            train_loss=loss_meter.value()[0],
            val_loss=val_loss
            #train_cm=str(confusion_matrix.value()),
            #val_cm=str(val_cm.value())
        ))

        #visualizer.plot('loss', loss_meter.value()[0])

        #print(loss_meter.value()[0] > pre_loss)

        if loss_meter.value()[0] > pre_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        pre_loss = loss_meter.value()[0]


# 记得改一下val loss的step，有错！
def val(model, dataloader):
    """
    obtain info from validation set
    """
    # settings
    criterion = torch.nn.L1Loss()
    visualizer = Visualizer(opt.env)
    vis = Visdom()
    model.eval()
    val_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #opt = DefaultConfig()
    val_loss_meter = meter.AverageValueMeter()
    #confusion_matrix = meter.ConfusionMeter(2)

    for batch_idx, data in tqdm(enumerate(dataloader)):
        print(data['img']['data'].shape)
        val_input = data['img']['data'].view(-1, 1, 190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)
        # deal with missing values
        val_input[torch.isnan(val_input)] = 0
        print("val input size: {}".format(val_input.shape))
        val_label = torch.tensor(list(map(float, data['target'])))  # tensor already?

        if opt.use_gpu:
            val_input = val_input.to(device)
            val_label = val_label.to(device)
            model = model.to(device)

        with torch.no_grad():
            val_res = model(val_input)

        #y = torch.zeros(opt.batch_size, 2)  # batch size, num_classes
        #y[torch.tensor(range(y.shape[0]), dtype=torch.long), val_label] = 1  # one-hot label

        #if opt.use_gpu:
        #    y = y.to(device)

        val_loss = criterion(val_res[0].squeeze(), val_label)  # predicted, label
        val_loss_meter.add(val_loss.data.cpu())

        #if batch_idx % opt.print_freq == 0:
        #visualizer.plot('val_loss', val_loss_meter.value()[0])

        predicted = val_res[0].squeeze()
        print("val prediction: {}".format(predicted))
        #confusion_matrix.add(predicted, val_label.data)

        val_step += 1
        vis.line([val_loss.item()], [val_step], win='val_loss', update='append')

        #confusion_matrix.add(val_res.detach().squeeze(), val_label.type(t.LongTensor))

    model.train()
    #cm_value = confusion_matrix.value()
    #accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return val_loss_meter.value()[0].cpu()



#test 看看数据啥情况再写！

def test(**kwargs):
    opt.parse(kwargs)
    # model
    model = Conv3d(num_classes=2).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    # data
    test_data_set = make_dataset(opt.test_imgs, labels=None)
    test_set = torchio.ImagesDataset(test_data_set.subjects)
    test_loader = DataLoader(test_set, batch_size=opt.test_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    results = []
    for batch_idx, data in enumerate(test_loader):
        print(data[opt.img_type]['data'].shape)
        input_data = data[opt.img_type]['data'].view(-1, 1, 192, 192,
                                                     120)  # tensor already? view(-1, 120, 192, 192, 1) org: torch.Size([2, 1, 192, 192, 120])
        input_data[torch.isnan(input_data)] = 0  # Nan all set to 0
        input_data = torch.autograd.Variable(input_data, volatile=True)
        targets = torch.tensor(list(map(float, data['label'])), dtype=torch.long)  # tensor already?
        if opt.use_gpu:
            input_data = input_data.to(device)
            res = model(input_data)
            print("last layer prob: {}".format(res[0].data))  # output from nn.sigmoid
            prob_1 = res[0].data[0][1]
            batch_res = [(targets_, prob_1) for targets_, prob_1_ in zip(targets, prob_1)]
            results += batch_res
    write_csv(results, opt.result_file)
    return results


if __name__ == "__main__":
    import fire
    fire.Fire()

