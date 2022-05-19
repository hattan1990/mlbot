from data.data_loader import EvalDataset, Dataset_BTC
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import CustomLoss

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import mlflow

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
        return model

    def _get_data(self, flag):
        args = self.args

        Data = Dataset_BTC
        timeenc = 0 if args.embed!='timeF' else 1

        shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
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
            feature_add=args.add_feature_num
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

    def vali(self, vali_data, vali_loader, criterion):
        def _check_strategy(pred, true):
            pred_hi = pred[:,:,0].detach().cpu()
            pred_lo = pred[:,:,1].detach().cpu()
            pred = (pred_hi+pred_lo)/2
            true_hi = true[:,:,0].detach().cpu()
            true_lo = true[:,:,1].detach().cpu()
            pred_min = pred.min(axis=1)[0]
            true_min = true_lo.min(axis=1)[0]
            pred_max = pred.max(axis=1)[0]
            true_max = true_hi.max(axis=1)[0]
            spread_loss = abs((pred_max - pred_min) - (true_max - true_min))
            diff_pred_max = abs(pred_max - true_max)
            diff_pred_min = abs(pred_min - true_min)

            acc = (pred_min > true_min)&(pred_max < true_max)&((pred_max-pred_min)>0)

            return spread_loss.mean(), diff_pred_max.mean(), diff_pred_min.mean(), acc.sum()/pred.shape[0]


        self.model.eval()
        total_loss = []
        total_loss_local = []
        total_spread_loss = []
        total_diff_pred_max = []
        total_diff_pred_min = []
        total_acc = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            if (batch_y.shape[1] == (self.args.label_len + self.args.pred_len)) & \
                    (batch_x.shape[1] == self.args.seq_len):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                loss_local = abs(pred.detach().cpu().numpy() - true.detach().cpu().numpy())
                total_loss.append(loss)
                total_loss_local.append(loss_local)
                spread_loss, diff_pred_max, diff_pred_min, acc = _check_strategy(pred, true)
                total_spread_loss.append(spread_loss)
                total_diff_pred_max.append(diff_pred_max)
                total_diff_pred_min.append(diff_pred_min)
                total_acc.append(acc)

        total_loss = np.average(total_loss)
        total_loss_local = np.average(total_loss_local)
        total_spread_loss = np.average(total_spread_loss)
        total_diff_pred_max = np.average(total_diff_pred_max)
        total_diff_pred_min = np.average(total_diff_pred_min)
        total_acc = np.average(total_acc)


        self.model.train()
        return total_loss, total_loss_local, total_spread_loss, total_diff_pred_max, total_diff_pred_min, total_acc

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
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                if (batch_y.shape[1] == (self.args.label_len + self.args.pred_len)) & \
                        (batch_x.shape[1] == self.args.seq_len):
                    iter_count += 1

                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
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

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_loss_local, spread_loss, diff_pred_max, diff_pred_min, acc = self.vali(vali_data, vali_loader, criterion)

            mlflow.log_metric("Cost time", int(time.time()-epoch_time), step=epoch + 1)
            mlflow.log_metric("Train Loss", train_loss, step=epoch + 1)
            mlflow.log_metric("Vali Loss", vali_loss, step=epoch + 1)
            mlflow.log_metric("ACC", acc, step=epoch + 1)
            mlflow.log_metric("Spread Diff", int(spread_loss), step=epoch + 1)
            mlflow.log_metric("Diff_pred_max", int(diff_pred_max), step=epoch + 1)
            mlflow.log_metric("Diff_pred_min", int(diff_pred_min), step=epoch + 1)
            mlflow.log_metric("Vali_loss local", int(vali_loss_local), step=epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ACC: {4:.7f} spread: {5} max: {6} min: {7}".format(
                epoch + 1, train_steps, train_loss, vali_loss, acc, int(spread_loss), int(diff_pred_max), int(diff_pred_min)))
            early_stopping(-acc, self.model, path)
            self.model.to(self.device)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model


    def predict(self, setting, load=True):
        def _check_mergin(raw):
            output = [raw['date'].max()]
            max_target = raw['hi'].max()
            min_target = raw['lo'].min()
            min_pred = raw['pred'].min()
            max_pred = raw['pred'].max()
            fix_values = [0, 1000, 2000]

            for fix in fix_values:
                if max_target > max_pred - fix:
                    check_max = 1
                else:
                    check_max = 0
                if min_target < min_pred + fix:
                    check_min = 1
                else:
                    check_min = 0
                if (check_max + check_min) == 2:
                    valid = 1
                else:
                    valid = 0
                spread = (max_pred - fix) - (min_pred + fix)
                output += [spread, check_max, check_min, valid]

            return output , [raw['date'].max(), max_target, min_target, max_pred, min_pred]
        args = self.args
        eval_data = EvalDataset(
            root_path=args.root_path,
            data_path = args.data_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            feature_add = args.add_feature_num
        )
        data_values, target_val, data_stamp, df_raw = eval_data.read_data()
        seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, seq_raw = eval_data.extract_data(data_values, target_val, data_stamp, df_raw)
        
        if load:
            best_model_path = 'checkpoint_cpu.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        cols = ['date', 'op', 'hi', 'lo', 'cl', 'volume']
        output = pd.DataFrame(columns = cols)
        spread_out1 = []
        spread_out2 = []
        for seq_x, seq_y, seq_x_mark, seq_y_mark, raw in\
                zip(seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, seq_raw):
            seq_x = torch.tensor(np.expand_dims(seq_x, axis=0))
            seq_y = torch.tensor(np.expand_dims(seq_y, axis=0))
            seq_x_mark = torch.tensor(np.expand_dims(seq_x_mark, axis=0))
            seq_y_mark = torch.tensor(np.expand_dims(seq_y_mark, axis=0))

            if seq_y.shape[1] == (self.args.label_len + self.args.pred_len):
                pred, _ = self._process_one_batch(
                    eval_data, seq_x, seq_y, seq_x_mark, seq_y_mark)

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

        return output.reset_index(), pd.DataFrame(spread_out1), pd.DataFrame(spread_out2, columns=['date', 't_max', 't_min', 'p_max', 'p_min'])

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
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

            return outputs, batch_y

        except:
            print("-------------------prediction error-------------------")
            print(batch_x.shape)
            print(batch_x_mark.shape)
            print(dec_inp.shape)
            print(batch_y_mark.shape)
            print("------------------------------------------------------")
            return None, None
