import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    customized loss function
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_left, output_right, label):
        eu_distance = F.pairwise_distance(output_left, output_right, keepdim=True)
        loss = torch.mean((1-label) * torch.pow(eu_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - eu_distance,
                                                                                                min=0.0), 2))
        return loss



class WeightedMSELoss(nn.Module):
    """
    customized loss function: weighted MSE
    """

    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight_70 = weight

    def transform_weights(self, weights):
        num_1 = sum(weights == self.weight_70)
        num_2 = sum(weights != self.weight_70)
        print(num_2,num_1)
        weights_transformed = torch.tensor(
            [weights[i] / num_1 if weights[i] == self.weight_70 else weights[i] / num_2 for i in
             range(weights.shape[0])])
        return weights_transformed

    def forward(self, targets, predicted):
        weights_init = torch.tensor([self.weight_70 if targets[i] >= 70 else 1-self.weight_70 for i in range(targets.shape[0])]).cuda()
        print(weights_init)
        weights = self.transform_weights(weights_init).cuda()
        print('weights:{}'.format(weights))
        loss = weights.expand_as(targets) * (targets - predicted)**2
        loss = torch.sum(loss)
        return loss

