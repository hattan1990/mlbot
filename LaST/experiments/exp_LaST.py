from utils.data_loader import *
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from utils.metrics import MAE, MSE
from models.LaST import LaST

from strategy import Estimation

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')


class Exp_LaST(Exp_Basic):
    def __init__(self, args):
        super(Exp_LaST, self).__init__(args)

    def _build_model(self):

        if self.args.features == 'S':
            in_dim, out_dim = 1, 1
            num_variables = 1
        elif self.args.features == 'MS':
            in_dim, out_dim = 8, 1
            num_variables = 1
        elif self.args.features == 'M':
            if "electricity" in self.args.data_path:
                in_dim, out_dim = 1, 1
                num_variables = 321
            elif "ETT" in self.args.data_path:
                in_dim, out_dim = 7, 7
                num_variables = 1
            elif "exchange_rate" in self.args.data_path:
                in_dim, out_dim = 1, 1
                num_variables = 8
            elif "weather" in self.args.data_path:
                in_dim, out_dim = 21, 21
                num_variables = 1
            elif "WTH" in self.args.data_path:
                in_dim, out_dim = 12, 12
                num_variables = 1
            else:
                raise Exception("'data_path' Error")
        else:
            raise Exception("KeyError: arg 'features' should be in ['S', 'M']")

        model = LaST(
            input_len=self.args.seq_len,
            output_len=self.args.pred_len,
            input_dim=in_dim,
            out_dim=out_dim,
            var_num=num_variables,
            latent_dim=self.args.latent_size,
            dropout=self.args.dropout, device=self._acquire_device())

        if self.args.devices == 'mps':
            return model
        else:
            return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'BTC':Dataset_BTC,
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            "Electricity": Dataset_Custom,
            "Exchange_rate": Dataset_Custom,
            'Weather': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            raise Exception("KeyError: Loss choice error. Please use word in ['mse', 'mae']")
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        mse_i, mses_i = [], []
        mae_i, maes_i = [], []
        estimation = Estimation(self.args)

        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(valid_loader):
            pred, pred_scale, true, true_scale, elbo, mlbo, mubo = self._process_one_batch_LaSTNet(
                valid_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            preds = pred_scale.detach().cpu().numpy()
            trues = true_scale.detach().cpu().numpy()

            masks = self._create_masks(pred, batch_eval)
            estimation.run_batch(index, pred, true, masks, batch_eval)

            mae_i.append(MAE(pred, true))
            maes_i.append(MAE(preds, trues))
            mse_i.append(MSE(pred, true))
            mses_i.append(MSE(preds, trues))

            total_loss.append(loss)

        total_loss = np.average(total_loss)

        mse = np.average(mse_i)
        mses = np.average(mses_i)
        mae = np.average(mae_i)
        maes = np.average(maes_i)
        rmse, rmses = np.sqrt(mse), np.sqrt(mses)

        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}'.format(mse, mae, rmse))
        print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}'.format(mses, maes, rmses))

        return total_loss, estimation

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data,
                                                     horizon=self.args.pred_len)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(train_loader):
                iter_count += 1
                for para_mode in range(2):
                    model_optim.zero_grad()

                    if para_mode == 0:
                        for para in self.model.parameters():
                            para.requires_grad = True

                        for para in self.model.LaSTLayer.MuboNet.parameters():
                            para.requires_grad = False
                        for para in self.model.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                            para.requires_grad = False
                        for para in self.model.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                            para.requires_grad = False

                    elif para_mode == 1:
                        for para in self.model.parameters():
                            para.requires_grad = False

                        for para in self.model.LaSTLayer.MuboNet.parameters():
                            para.requires_grad = True
                        for para in self.model.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                            para.requires_grad = True
                        for para in self.model.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                            para.requires_grad = True

                    pred, pred_scale, true, true_scale, elbo, mlbo, mubo = self._process_one_batch_LaSTNet(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = (criterion(pred, true) - elbo - mlbo + mubo) if para_mode == 0 else (mubo - mlbo)

                    loss.backward()
                    model_optim.step()

                    train_loss.append(loss.item())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            target_time_range_from = 2000000
            target_time_range_to = 6000000

            print('--------start to validate-----------')
            valid_loss, estimation = self.valid(valid_data, valid_loader, criterion)
            acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, acc4_ex, cnt12, values12, dict12, strategy_data = estimation.run(
                epoch, target_time_range_from, target_time_range_to)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss))

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch + 1, self.args)
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion(self.args.loss)

        self.model.eval()

        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        loss, estimation = self.valid(test_data, test_loader, criterion)
        target_time_range_from = 2000000
        target_time_range_to = 6000000
        estimation.run(100, target_time_range_from, target_time_range_to)

        return loss

    def _process_one_batch_LaSTNet(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        if self.args.devices == 'mps':
            batch_x = batch_x.to(torch.float32).to(self.model.device)
            batch_y = batch_y.to(torch.float32)
        else:
            batch_x = batch_x.double().to(self.model.device)
            batch_y = batch_y.double()

        outputs, elbo, mlbo, mubo = self.model(batch_x)

        outputs_scaled = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.model.device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        return outputs, outputs_scaled, batch_y, batch_y_scaled, elbo, mlbo, mubo

    def _create_masks(self, batch_y, batch_val, mergin=20000):
        masks = []
        for hi_lo, val in zip(batch_y, batch_val):
            op = val[0, 1]
            hi_max = hi_lo[:, 0].max()
            lo_min = hi_lo[:, 0].min()
            spread1 = (hi_max - op)
            spread2 = (op - lo_min)
            if (spread1 >= mergin) or (spread2 >= mergin):
                masks.append(True)
            else:
                masks.append(False)

        return torch.tensor(masks)