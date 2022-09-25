from data.data_loader import Dataset_ETT_hour, Dataset_BTC, Dataset_Custom, Dataset_Pred, Dataset_ECL_hour
from exp.exp_basic import Exp_Basic
from models.model import Informer, Yformer, Yformer_skipless

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

from strategy import Estimation

import warnings

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'yformer': Yformer,
            'yformer_skipless': Yformer_skipless

        }
        if self.args.model == 'informer' or self.args.model == "yformer" or self.args.model == "yformer_skipless":
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'BTC': Dataset_BTC,
            'ETTm2': Dataset_BTC,
            'ECL': Dataset_ECL_hour,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            use_decoder_tokens=args.use_decoder_tokens,
            timeenc=timeenc,
            freq=freq
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _inverse_transform_batch(self, batch_values, scaler):
        output = []
        for values in batch_values:
            out = scaler.inverse_transform(values)
            output.append(out)
        return np.array(output)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        total_loss_real = []
        estimation = Estimation(self.args)
        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = torch.tensor(batch_y[:, -self.args.pred_len:, f_dim:], dtype=torch.float32).to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            pred_real = self._inverse_transform_batch(pred.numpy(), vali_data.scaler_target)
            true_real = self._inverse_transform_batch(true.numpy(), vali_data.scaler_target)

            loss = criterion(pred, true)
            loss_real = np.mean(abs(pred_real - true_real))

            total_loss.append(loss)
            total_loss_real.append(loss_real)
            # Strategyモジュール追加
            batch_eval = batch_eval[:, -self.args.pred_len:, :].to(self.device)
            masks = self._create_masks(pred_real, batch_eval)
            estimation.run_batch(index, pred_real, true_real, masks, batch_eval)

        total_loss = np.average(total_loss)
        total_loss_real = np.average(total_loss_real)
        self.model.train()
        return total_loss, total_loss_real, estimation

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            auto_train_loss = []
            combined_train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                if self.args.use_decoder_tokens:
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = dec_inp.float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = torch.tensor(batch_y[:, -self.args.pred_len:, f_dim:], dtype=torch.float32).to(self.device)

                auto_loss = criterion(outputs[:, :-self.args.pred_len, :], batch_x)
                auto_train_loss.append(auto_loss.item())
                loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y)
                train_loss.append(loss.item())

                combined_loss = self.args.alpha * auto_loss + (1 - self.args.alpha) * loss
                combined_train_loss.append(combined_loss.item())
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            auto_loss = np.average(auto_train_loss)
            combined_loss = np.average(combined_train_loss)
            vali_loss, vali_loss_real, estimation = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Auto Loss : {3:.7f} | Comb Loss : {4:.7f}, Vali Loss: {5:.7f}, Vali Loss Real: {6:.7f}".format(
                    epoch + 1, train_steps, train_loss, auto_loss, combined_loss, vali_loss, vali_loss_real))
            estimation.run(epoch)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = torch.tensor(batch_y[:, -self.args.pred_len:, f_dim:], dtype=torch.float32).to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = torch.tensor(batch_y[:, -self.args.pred_len:, f_dim:], dtype=torch.float32).to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _create_masks(self, batch_y, batch_val, mergin=20000):
        masks = []
        for hi_lo, val in zip(batch_y, batch_val):
            op = val[0, 1]
            hi_max = hi_lo[: ,0].max()
            lo_min = hi_lo[: ,0].min()
            spread1 = (hi_max - op)
            spread2 = (op - lo_min)
            if (spread1 >= mergin) or (spread2 >= mergin):
                masks.append(True)
            else:
                masks.append(False)

        return torch.tensor(masks)