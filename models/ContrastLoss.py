import torch
import torch.nn as nn
import torch.nn.functional as F


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