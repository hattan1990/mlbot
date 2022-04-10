import numpy as np
from torch import nn
import torch

class CustomLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()
        # パラメータを設定
        self.mode = mode

    def forward(self, pred, true):
        if self.mode == 'penalties':
            pred1 = pred[:, :, 0]
            pred2 = pred[:, :, 1]
            true1 = true[:, :, 0]
            true2 = true[:, :, 1]
            diff1 = true1 - pred1
            diff2 = pred2 - true2

            check_diff1 = diff1 < 0
            diff1[check_diff1] *= 2.0
            diff1[~check_diff1] *= 0.5
            check_diff2 = diff2 < 0
            diff2[check_diff2] *= 2.0
            diff2[~check_diff2] *= 0.5
            loss = torch.mean(torch.abs(diff1)) + torch.mean(torch.abs(diff2))

        elif self.mode == 'min_max':
            min_pred = pred.min()
            max_pred = pred.max()
            min_true = true.min()
            max_true = true.max()
            loss_min = torch.abs(min_pred - min_true)
            loss_max = torch.abs(max_pred - max_true)
            loss = loss_min + loss_max
        else:
            loss = torch.mean(torch.abs(true - pred))


        return loss