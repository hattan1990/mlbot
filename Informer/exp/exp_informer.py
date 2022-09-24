from data.data_loader import EvalDataset, Dataset_BTC
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import CustomLoss

from strategy import Estimation

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
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
                e_layers, # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args['load_models'] == True:
            best_model_path = 'best_model.pth'
            model.load_state_dict(torch.load(best_model_path))
            print("load model weights : {}".format(best_model_path))
        return model

    def _get_data(self, flag):
        args = self.args

        Data = Dataset_BTC
        timeenc = 0 if args.embed!='timeF' else 1

        if (flag == 'val') or (flag == 'test'):
            eval_mode = True
            shuffle_flag = False
        else:
            eval_mode = False
            shuffle_flag = True

        drop_last = True; batch_size = args.batch_size; freq=args.freq
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
            feature_add=args.add_feature_num,
            option=args.data_option,
            eval_mode=eval_mode
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = CustomLoss(self.args.loss_mode)
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
            pred, true, masks, val = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval)

            batch_y = torch.tensor(batch_y[:, -self.args.pred_len:, :], dtype=torch.float32).to(self.device)
            true = batch_y.detach().cpu()
            pred = pred.detach().cpu()
            val = val.detach().cpu()

            pred_real = self._inverse_transform_batch(pred.numpy(), vali_data.scaler_target)
            true_real = self._inverse_transform_batch(true.numpy(), vali_data.scaler_target)

            loss = criterion(pred, true)
            loss_real = np.mean(abs(pred_real - true_real))

            total_loss.append(loss)
            total_loss_real.append(loss_real)
            #Strategyモジュール追加
            estimation.run_batch(index, pred_real, true_real, masks, val)
        total_loss = np.average(total_loss)
        total_loss_real = np.average(total_loss_real)
        self.model.train()
        return total_loss, total_loss_real, estimation


    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_val) in enumerate(train_loader):
                if (batch_y.shape[1] == (self.args.label_len + self.args.pred_len)) & \
                        (batch_x.shape[1] == self.args.seq_len):
                    iter_count += 1

                    model_optim.zero_grad()
                    pred, true, masks, _ = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_val)

                    if self.args['target'] is None:
                        num = self.args['target_num']
                        loss = criterion(pred, true[:,:,num].unsqueeze(2))
                    else:
                        loss = criterion(pred, true)
                        train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    break


            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_loss_real, estimation = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss : {3:.7f} | Vali Loss Real: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss,  vali_loss, vali_loss_real))

            estimation.run(epoch)

            early_stopping(vali_loss, self.model, path)
            self.model.to(self.device)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.args.best_score


    def predict(self, load=True):
        def _check_mergin(raw):
            output = [raw['date'].max()]
            max_target = raw['hi'].max()
            min_target = raw['lo'].min()

            min_pred = raw['pred'].min()
            max_pred = raw['pred'].max()

            fix_sp = (raw['pred_hi'].max() - raw['pred_lo'].min()) / 4
            min_pred2 = raw['pred_lo'].min() + fix_sp
            max_pred2 = raw['pred_hi'].max() - fix_sp
            min_pred3 = raw['pred_lo'].mean()
            max_pred3 = raw['pred_hi'].mean()

            preds_list = [[max_pred, min_pred], [max_pred2, min_pred2], [max_pred3, min_pred3]]

            for max_p, min_p in preds_list:
                if max_target > max_p:
                    check_max = 1
                else:
                    check_max = 0
                if min_target < min_p:
                    check_min = 1
                else:
                    check_min = 0
                if (check_max + check_min) == 2:
                    valid = 1
                else:
                    valid = 0
                spread = (max_p) - (min_p)
                output += [spread, check_max, check_min, valid]

            out_sp = max_pred2 - min_pred2
            max_check = max_pred2 < max_target
            min_check = min_pred2 > min_target
            return output , [raw['date'].max(), max_target, min_target, max_pred2, min_pred2, out_sp, max_check, min_check]

        args = self.args
        eval_data = EvalDataset(
            root_path=args.root_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.eval_data,
            target=args.target,
            inverse=args.inverse,
            feature_add = args.add_feature_num,
            option=args.data_option
        )
        data_values, target_val, data_stamp, df_raw = eval_data.read_data()
        road_data = eval_data.extract_data(data_values, target_val, data_stamp, df_raw)

        if load:
            seq_len = str(args.seq_len)
            label_len = str(args.label_len)
            pred_len = str(args.pred_len)
            n_heads = str(args.n_heads)
            best_model_path = 'weights/' + seq_len + '_' + label_len +'_' + pred_len + '_' + n_heads + '.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print("load trained model {}".format(best_model_path))

        self.model.eval()

        cols = ['date', 'op', 'hi', 'lo', 'cl', 'volume']
        output = pd.DataFrame(columns = cols)
        spread_out1 = []
        spread_out2 = []
        for seq_x, seq_y, seq_x_mark, seq_y_mark, raw in tqdm(road_data):
            seq_x = torch.tensor(np.expand_dims(seq_x, axis=0))
            seq_y = torch.tensor(np.expand_dims(seq_y, axis=0))
            seq_x_mark = torch.tensor(np.expand_dims(seq_x_mark, axis=0))
            seq_y_mark = torch.tensor(np.expand_dims(seq_y_mark, axis=0))

            if seq_y.shape[1] == (self.args.label_len + self.args.pred_len):
                pred, _, _, _ = self._process_one_batch(
                    eval_data, seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y)

                pred_hi = pred[0, :, 0].detach().cpu().numpy()
                pred_lo = pred[0, :, 1].detach().cpu().numpy()
                raw = pd.DataFrame(raw[-self.args.pred_len:, :6], columns=cols)
                raw['pred_hi'] = pred_hi
                raw['pred_lo'] = pred_lo
                raw['pred'] = (raw['pred_hi'] + raw['pred_lo'])/2
                output = pd.concat([output, raw])
                out1, out2 = _check_mergin(raw)
                spread_out1.append(out1)
                spread_out2.append(out2)

        return output.reset_index(), pd.DataFrame(spread_out1), pd.DataFrame(spread_out2, columns=['date', 't_max', 't_min', 'p_max', 'p_min', 'spread', 'max_tf', 'min_tf'])

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

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_val):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        try:
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
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            batch_val = batch_val[:, -self.args.pred_len:, f_dim:].to(self.device)

            masks = self._create_masks(outputs, batch_val)

            return outputs, batch_y, masks, batch_val

        except:
            print("-------------------prediction error-------------------")
            print(batch_x.shape)
            print(batch_x_mark.shape)
            print(dec_inp.shape)
            print(batch_y_mark.shape)
            print("------------------------------------------------------")
            return None, None
