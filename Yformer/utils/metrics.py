import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

class CustomLoss(torch.nn.Module):
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
            diff1[check_diff1] *= 1.11
            diff1[~check_diff1] *= 0.99
            check_diff2 = diff2 < 0
            diff2[check_diff2] *= 1.11
            diff2[~check_diff2] *= 0.99
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