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
from torch.autograd import Variable
import numpy as np
from visdom import Visdom
from utils.visualize import Visualizer
from tqdm import tqdm


#img_path = './data/img_data/'
#labels = ['1', '0']  # would be changed later by the exact formats of labels
#imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]


def train():
    # model
    #opt = DefaultConfig()
    global_step = 0
    model = Conv3d(num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay=opt.weight_decay)
    # statistical results
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    pre_loss = 1e10

    # create training and validation set
    data_set = make_dataset(opt.train_imgs, opt.train_labels)
    num_subjects = len(data_set)
    training_split_ratio = opt.training_split_ratio
    num_training_subjects = int(training_split_ratio * num_subjects)

    training_subjects = data_set.subjects[:num_training_subjects]
    validation_subjects = data_set.subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(training_subjects)
    validation_set = torchio.ImagesDataset(validation_subjects)

    training_loader = DataLoader(training_set, batch_size=opt.batch_size)
    val_loader = DataLoader(validation_set, batch_size=opt.batch_size)

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

            # check shapes
            print(data[opt.img_type]['data'].shape)

            # org: torch.Size([2, 1, 192, 192, 120]), we want depth = 120, channel = 1
            input_data = data[opt.img_type]['data'].view(-1, 1, 192, 192, 120)

            # check one slice
            vis_data_1 = (data['FA']['data'][0, 0, :, :, :][:, :, 69]) * 255.0
            vis_data_2 = (data['FA']['data'][0, 0, :, :, :][69, :, :]) * 255.0
            vis_data_3 = (data['FA']['data'][0, 0, :, :, :][:, 69, :]) * 255.0

            # deal with missing values
            input_data[torch.isnan(input_data)] = 0  # Nan all set to 0

            # get labels
            targets = torch.tensor(list(map(float, data['label'])), dtype=torch.long)  # tensor already?

            # check data and the shapes: targets, input_size, target_size, one_slice_shape
            print("targets : {}".format(targets))
            print("input size: {}".format(input_data.shape))
            print("target size: {}".format(targets.shape))
            print(input_data[0, :, :, :, :].view(120, 192, 192)[89, :, :].shape)

            # check the images from one slice
            visualizer.img("slice {}".format("69_1"), vis_data_1)
            visualizer.img("slice {}".format("69_2"), vis_data_2.T)
            visualizer.img("slice {}".format("69_3"), vis_data_3)

            # training
            if opt.use_gpu:
                input_data = input_data.to(device)
                targets = torch.tensor(targets).to(device)

            optimizer.zero_grad()
            res = model(input_data)  # a list, res[0].data: prob from output layer; res[1]: x_1
            print("last layer prob: {}".format(res[0].data))
            loss = criterion(res[0], targets)
            print("train_loss: {}".format(loss.data))

            loss.backward()
            optimizer.step()
            global_step += 1

            vis.line([loss.item()], [global_step], win='train_loss', update='append')

            loss_meter.add(loss.data)
            predicted = torch.max(res[0].data, 1)[1]
            confusion_matrix.add(predicted, targets.data)

            #print(confusion_matrix)

            if batch_idx % opt.print_freq == 0:
                visualizer.plot('train_loss', loss_meter.value()[0])  # loss_meter.value()[0]: mean value of loss

        if epoch % opt.print_freq == 0:
            print("Epoch: {}/{} | loss: {}".format(epoch, opt.max_epoch, loss.item()))


        #model.save()

        # validation
        val_cm, val_acc = val(model, val_loader)  # val_cm: confusion matrix for validation; val_acc: accuracy

        visualizer.plot('val_acc', val_acc)
        visualizer.log("epoch:{epoch}    lr: {lr}    train_loss: {train_loss}    train_cm:{train_cm}    val_cm:{val_cm}".format(
            epoch=epoch,
            lr=opt.lr,
            train_loss=loss_meter.value()[0],
            train_cm=str(confusion_matrix.value()),
            val_cm=str(val_cm.value())
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
    criterion = torch.nn.CrossEntropyLoss()
    visualizer = Visualizer(opt.env)
    vis = Visdom()
    model.eval()
    val_step = 0

    #opt = DefaultConfig()
    val_loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    for batch_idx, data in tqdm(enumerate(dataloader)):
        val_input = data[opt.img_type]['data'].view(-1, 1, 192, 192, 120)  # tensor already?
        val_input[torch.isnan(val_input)] = 0
        val_label = torch.tensor(list(map(float, data['label'])), dtype=torch.long)  # tensor already?

        if opt.use_gpu:
            val_input.to(device)
            val_label.to(device)

        val_res = model(val_input)
        val_loss = criterion(val_res[0], val_label)
        val_loss_meter.add(val_loss.data)

        #if batch_idx % opt.print_freq == 0:
        #visualizer.plot('val_loss', val_loss_meter.value()[0])

        predicted = torch.max(val_res[0].data, 1)[1]
        confusion_matrix.add(predicted, val_label.data)

        val_step += 1
        vis.line([val_loss.item()], [val_step], win='val_loss', update='append')

        #confusion_matrix.add(val_res.detach().squeeze(), val_label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy



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

