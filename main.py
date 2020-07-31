# -*- coding: utf-8 -*-
# @Time    : 2020-07-24 16:06
# @Author  : Zongchao Liu
# @FileName: main.py
# @Software: PyCharm


import torch
from config import DefaultConfig
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
    opt = DefaultConfig()
    global_step = 0
    model = Conv3d(num_classes = 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay=opt.weight_decay)
    # statistical results
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    pre_loss = 1e10

    # create training and val
    data_set = make_dataset(opt.imgs, opt.labels)
    num_subjects = len(data_set)
    training_split_ratio = opt.training_split_ratio
    num_training_subjects = int(training_split_ratio * num_subjects)
    subjects = data_set.subjects

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(training_subjects)
    validation_set = torchio.ImagesDataset(validation_subjects)

    training_loader = DataLoader(training_set, batch_size = opt.batch_size)
    val_loader = DataLoader(validation_set, batch_size = opt.batch_size)

    # visualization

    vis = Visdom()
    visualizer = Visualizer(opt.env)
    vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))


    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for batch_idx, data in enumerate(training_loader):

            input_data = (data[opt.img_type]['data'].view(-1, 120, 192, 192, 1))/255.0  # tensor already?
            input_data[torch.isnan(input_data)] = 0  # Nan all set to 0
            targets = torch.tensor(list(map(float, data['label'])), dtype=torch.long)  # tensor already?
            print(targets)
            print("input size:{}".format(input_data.shape))
            print("target size:{}".format(targets.shape))
            if opt.use_gpu:
                input_data = input_data.to(device)
                targets = torch.tensor(targets).to(device)

            optimizer.zero_grad()
            res = model(input_data)
            #print(res[0])
            loss = criterion(res[0], targets)
            print(loss.data)
            loss.backward()
            optimizer.step()
            global_step += 1
            vis.line([loss.item()], [global_step], win='train_loss', update='append')

            loss_meter.add(loss.data)
            print("last layer prob: {}".format(res[0].data))  # output from nn.sigmoid
            predicted = torch.max(res[0].data, 1)[1]
            confusion_matrix.add(predicted, targets.data)
            #print(confusion_matrix)

            if batch_idx % opt.print_freq == 0:
                visualizer.plot('loss', loss_meter.value()[0])


        if epoch % opt.print_freq == 0:
            print("Epoch: {}/{} | loss: {}".format(epoch, opt.max_epoch, loss.item()))


        #model.save()

        # validation
        val_cm, val_acc = val(model, val_loader)
        visualizer.plot('val_acc', val_acc)
        visualizer.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}".format(
            epoch = epoch,
            lr=opt.lr,
            loss = loss_meter.value()[0],
            train_cm = str(confusion_matrix.value()),
            val_cm = str(val_cm.value())
        ))

        #print(loss_meter.value()[0] > pre_loss)

        if loss_meter.value()[0] > pre_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        pre_loss = loss_meter.value()[0]


def val(model,dataloader):
    """
    obtain info from validation set
    """
    model.eval()
    opt = DefaultConfig()
    confusion_matrix = meter.ConfusionMeter(2)
    for batch_idx, data in tqdm(enumerate(dataloader)):
        val_input = data[opt.img_type]['data'].view(-1, 120, 192, 192, 1)  # tensor already?
        val_input[torch.isnan(val_input)] = 0
        val_label = torch.tensor(list(map(float, data['label'])), dtype=torch.long)  # tensor already?

        if opt.use_gpu:
            val_input.to(device)
            val_label.to(device)

        val_res = model(val_input)
        predicted = torch.max(val_res[0].data, 1)[1]
        confusion_matrix.add(predicted, val_label.data)
        #confusion_matrix.add(val_res.detach().squeeze(), val_label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy



#test 看看数据啥情况再写！

#def test(**kwargs):
#    opt.parse(kwargs)
#
#    if opt.load_model_path:
#        model.load(opt.load_model_path)
#    if opt.use_gpu:
#        model.to(device)
#
#    # data
#    test_data =




def help():
    print('help')


if __name__ == "__main__":
    import fire
    fire.Fire()

