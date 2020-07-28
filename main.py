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


#img_path = './data/img_data/'
#labels = ['1', '0']  # would be changed later by the exact formats of labels
#imgs = [os.path.join(img_path, img) for img in sorted(os.listdir(img_path))[1:]]


opt = DefaultConfig()
lr = opt.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for _, data in enumerate(training_loader):

            input = data[opt.img_type]['data'].reshape(opt.batch_size,192,192,120) # tensor already?
            target = data['label'] # tensor already?
            if opt.use_gpu:
                input = input.cuda
                target = target.cuda()
            optimizer.zero_grad()
            res = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data[0])
            confusion_matrix.add(res.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        model.save()

        val_cm, val_acc = val(model, val_dataloader)









  def val(model, dataloader):
    pass

  def test():
    pass

  def help():
    print('help')


if __name__ == "__main__":
    import fire
    fire.Fire()

