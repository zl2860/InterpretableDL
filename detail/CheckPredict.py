import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from data.dataset import make_dataset
import torchio
from config import opt
from torch.utils.data import DataLoader
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


def get_predict(model, dataloader):
    """
    get predicted results
    """
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict = []
    id = []
    for batch_idx, data in enumerate(dataloader):
        print(data['img']['data'].shape)
        print(data['target'].shape)
        val_input = data['img']['data'].view(-1, 1, 190-opt.crop_size*2, 190-opt.crop_size*2, 190-opt.crop_size*2)
        val_id = data['id']
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
        predict.append(val_res[0].squeeze().cpu())
        id.append(val_id)
    model.train()
    return predict, id


def flat(l, tensor=True):
    flatten = []
    for sub_list in l:
        for idx in range(len(sub_list)):
            if tensor:
                flatten.append((sub_list[idx].type(torch.int)).cpu().item())
            else:
                flatten.append((sub_list[idx]))
    return flatten


# get true validation scores
def get_true_val(dataloader):
    true_score = []
    true_id = []
    for batch_idx, data in enumerate(dataloader):
        val_label = torch.tensor(list(map(float, data['target'])))
        val_id = data['id']
        for idx in range(len(val_label)):
            true_score.append(val_label[idx].cpu().item())
            true_id.append(val_id[idx])
    print("length: {}".format(len(true_score)))
    return true_score, true_id


if __name__ == "__main__":
    # get predicted results
    from data.dataset import *
    from config import opt
    import time
    training_loader, val_loader = create_train_val(target='score')
    model = torch.load('./checkpoints/exp_score_09-10.pth')  # your trained model's path

    # get validation results
    val_predict, val_id = get_predict(model, val_loader)
    val_id = flat(val_id, tensor=False)

    val_res = flat(val_predict)
    val_true, val_true_id = get_true_val(val_loader)

    df_val_predict = pd.DataFrame({"val_predict": val_res, "id": val_id})
    df_val_predict.to_csv("prediction_val{}.csv".format(time.strftime('%m-%d')))

    df_val_true = pd.DataFrame({"val_true": val_true, "id": val_true_id})
    df_val_true.to_csv("true_val{}.csv".format(time.strftime('%m-%d')))

    # get training results
    train_predict, train_id = get_predict(model, training_loader)
    train_id = flat(train_id, tensor=False)

    train_res = flat(train_predict)
    train_true, train_true_id = get_true_val(training_loader)

    df_train_predict = pd.DataFrame({"val_predict": train_res, "id": train_id})
    df_train_predict.to_csv("prediction_train{}.csv".format(time.strftime('%m-%d')))

    df_train_true = pd.DataFrame({"val_true": train_true, "id": train_true_id})
    df_train_true.to_csv("true_train{}.csv".format(time.strftime('%m-%d')))

    #bins = np.linspace(50, 100, 20)
    #
    #plt.hist(val_res, bins, alpha=0.5, label='predicted attention score')
    #plt.hist(train_ture, bins, alpha=0.5, label='true attention score')
    #plt.legend(loc='upper right')
    #plt.show()
    #
    #plt.scatter(x=val_ture, y=val_res, alpha=0.5)
    #plt.xlabel('True attention score')
    #plt.ylabel('Predicted attention score')
    #plt.legend(loc='upper right')
    #plt.show()
    #
    #sns.regplot(val_res, val_ture)
    #plt.show()