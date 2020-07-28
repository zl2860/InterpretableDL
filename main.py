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
#from torch.autograd import Variable
from visdom import Visdom
from utils.visualize import Visualizer


#img_path = './data/img_data/'
#labels = ['1', '0']  # would be changed later by the exact formats of labels
#imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]


opt = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global_step = 0

def train():
    # model
    model = Conv3d(num_classes = 2)

    # data preparation
    training_loader = DataLoader(training_set)

    # loss
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay=opt.weight_decay)
    # statistical results
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    pre_loss = 1e100

    # create training and val
    data_set = make_dataset(opt.imgs, opt.labels)
    num_subjects = len(data_set)
    training_split_ratio = 0.8
    num_training_subjects = int(training_split_ratio * num_subjects)
    subjects = data_set.subjects

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(training_subjects)
    validation_set = torchio.ImagesDataset(validation_subjects)

    training_loader = DataLoader(training_set, batch_size= opt.batch_size)
    val_loader = DataLoader(validation_set, batch_size= opt.batch_size)

    # visualization

    vis = Visdom()
    visualizer = Visualizer(opt.env)
    vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))


    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for batch_idx, data in enumerate(training_loader):

            input = data[opt.img_type]['data'].reshape(opt.batch_size,192,192,120) # tensor already?
            target = data['label'] # tensor already?
            if opt.use_gpu:
                inputs = input.to(device)
                targets = target.to(device)
            optimizer.zero_grad()
            res = model(inputs)
            loss = criterion(score,targets)
            loss.backward()
            optimizer.step()
            global_step += 1
            vis.line([loss.item()], [global_step], win='train_loss', update='append')

            loss_meter.add(loss.data[0])
            confusion_matrix.add(res.data, target.data)

            if batch_idx % opt.print_freq == 0:
                visualizer.plot('loss', loss_meter.value()[0])


        if epoch % opt.print_freq == 0:
            print("Epoch: {}/{} | loss: {}".format(epoch, opt.max_epoch, loss.item()))


        model.save()

        # validation
        val_cm, val_acc = val(model, val_dataloader)
        visualizer.plot('val_acc', val_acc)
        visualizer.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm), val_cm:{val_cm}".format(
            epoch = epoch,
            loss = loss_meter.value()[0]
            val_cm = str(val_cm.value()),
            train_cm = str(confusion_matrix.value()),
            lr = lr
        ))

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
    confusion_matrix = meter.ConfusionMeter(2)
    for batch_idx, data in tqdm(enumerate(dataloader)):
        val_input = data[opt.img_type]['data'].reshape(opt.batch_size, 192, 192, 120)  # tensor already?
        val_label = data['label'] # tensor already?

        if opt.use_gpu:
            val_input.to(device)
            val_label.to(device)

        val_res = model(val_input)
        confusion_matrix.add(val_res.detach().squeeze(), val_label.type(t.LongTensor))

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
#        model.cuda()
#
#    # data
#    test_data =




def help():
    print('help')


if __name__ == "__main__":
    import fire
    fire.Fire()

