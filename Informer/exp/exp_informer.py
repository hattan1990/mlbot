from data.data_loader import EvalDataset, Dataset_BTC
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import CustomLoss

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
            seq_len = str(self.args.seq_len)
            label_len = str(self.args.label_len)
            pred_len = str(self.args.pred_len)
            n_heads = str(self.args.n_heads)
            best_model_path = 'weights/' + seq_len + '_' + label_len +'_' + pred_len + '_' + n_heads + '.pth'
            model.load_state_dict(torch.load(best_model_path))
            print("load model weights : {}".format(best_model_path))
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
            feature_add=args.add_feature_num,
            option=args.data_option
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
        def _create_tmp_data(pred_data, true_data, val_data):
            output = pd.DataFrame()
            for true, pred, val in zip(true_data, pred_data, val_data):
                true = true.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                val = val.detach().cpu().numpy()
                tmp_values = np.concatenate([val, true, pred], axis=1) * 10000000
                columns = ['date', 'op', 'cl', 'hi', 'lo', 'pred_hi', 'pred_lo']

                tmp_data = pd.DataFrame(tmp_values, columns=columns)
                output = pd.concat([output, tmp_data])

            return output

        def _back_test_spot_swing(trade_data, threshold=15000, pred_opsion=''):
            output = []
            total = 0
            trade_cnt = 0
            for i in range(0, trade_data.shape[0], 12):
                if i == 0:
                    start = 0
                    end = 12 - 1
                else:
                    end = i + 12 - 1

                tmp_data = trade_data.loc[start:end]
                base_price = tmp_data['op'].values[0]
                if pred_opsion == 'mean':
                    pred_spread_min = int(tmp_data['pred_lo'].mean())
                    pred_spread_max = int(tmp_data['pred_hi'].mean())
                elif pred_opsion == 'min_max':
                    spread = tmp_data['pred_hi'].max() - tmp_data['pred_lo'].min()
                    pred_spread_min = int(tmp_data['pred_lo'].min() + spread / 4)
                    pred_spread_max = int(tmp_data['pred_hi'].max() - spread / 4)

                elif pred_opsion == 'zero':
                    pred_spread_min = int(tmp_data['pred_lo'].values[0])
                    pred_spread_max = int(tmp_data['pred_hi'].values[0])
                else:
                    pred_spread_min = int(tmp_data['pred'].min())
                    pred_spread_max = int(tmp_data['pred'].max())
                spread_to_max = (pred_spread_max - base_price)
                spread_to_min = (base_price - pred_spread_min)

                buy = False
                sell = False

                if (spread_to_max >= threshold) & (spread_to_max > spread_to_min):
                    trade_cnt += 1
                    buy = True
                    buy_price = base_price
                    for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                        if hi > pred_spread_max:
                            sell = True
                            sell_price = pred_spread_max

                elif (spread_to_min >= threshold) & (spread_to_max < spread_to_min):
                    trade_cnt += 1
                    sell = True
                    sell_price = base_price
                    for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                        if lo < pred_spread_min:
                            buy = True
                            buy_price = pred_spread_min

                close_price = tmp_data['cl'].values[-1]
                close_date = tmp_data['date'].values[-1]

                if (sell == True) & (buy == True):
                    profit = sell_price - buy_price
                elif (sell == True) & (buy == False):
                    profit = sell_price - close_price
                elif (sell == False) & (buy == True):
                    profit = close_price - buy_price
                else:
                    profit = 0
                    pass
                total += profit
                output.append([close_date, total, profit, buy, sell])
                start = end + 1

            output = pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell'])
            output = output[output['profit'] != 0]
            term = (output['buy'] == True) & (output['sell'] == True)
            profit_win = output.loc[term, 'profit'].sum()
            profit_loss = output.loc[~term, 'profit'].sum()

            return output, (total, profit_win, profit_loss)

        def _check_strategy(pred_data, true, val):
            tmp_out = _create_tmp_data(pred_data, true, val)
            #profit_min_max = _back_test_spot_swing(pred_data, true, val, threshold=15000, pred_opsion='min_max')
            #profit_mean = _back_test_spot_swing(pred_data, true, val, threshold=15000, pred_opsion='mean')

            pred_hi = pred_data[:,:,0].detach().cpu() * 10000000
            pred_lo = pred_data[:,:,1].detach().cpu() * 10000000
            pred = (pred_hi+pred_lo)/2
            true_hi = true[:,:,0].detach().cpu() * 10000000
            true_lo = true[:,:,1].detach().cpu() * 10000000

            pred_min = pred.min(axis=1)[0]
            pred_min2 = pred_lo.min(axis=1)[0]
            pred_min3 = pred_lo.mean(axis=1)
            true_min = true_lo.min(axis=1)[0]

            pred_max = pred.max(axis=1)[0]
            pred_max2 = pred_hi.min(axis=1)[0]
            pred_max3 = pred_hi.mean(axis=1)
            true_max = true_hi.max(axis=1)[0]

            spread_4 = (pred_max2 - pred_min2) / 4

            acc1 = (pred_min > true_min) & (pred_max < true_max) & ((pred_max - pred_min) > 0)
            acc2 = ((pred_min2 + spread_4) > true_min) & ((pred_max2 - spread_4) < true_max) & ((pred_max2 - pred_min2) > 0)
            acc3 = (pred_min3 > true_min) & (pred_max3 < true_max) & ((pred_max3 - pred_min3) > 0)

            return acc1.sum()/pred.shape[0], acc2.sum()/pred.shape[0], acc3.sum()/pred.shape[0], tmp_out


        self.model.eval()
        total_loss = []
        total_loss_ex = []
        total_loss_local = []
        total_acc1 = []
        total_acc1_ex = []
        total_acc2 = []
        total_acc2_ex = []
        total_acc3 = []
        total_acc3_ex = []
        total_profit_min_max = np.array([0.0, 0.0, 0.0])
        total_profit_mean = np.array([0.0, 0.0, 0.0])
        ex_count = 0
        strategy_data = pd.DataFrame()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_val) in enumerate(vali_loader):
            if (batch_y.shape[1] == (self.args.label_len + self.args.pred_len)) & \
                    (batch_x.shape[1] == self.args.seq_len):
                pred, true, masks, val = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_val)

                if self.args['extra'] == True:
                    pred_ex = pred[masks]
                    true_ex = true[masks]
                    val_ex = val[masks]
                    if true_ex.shape[0] > 0:
                        loss_ex = criterion(pred_ex.detach().cpu(), true_ex.detach().cpu())
                        total_loss_ex.append(loss_ex)
                        acc1_ex, acc2_ex, acc3_ex, tmp_out = _check_strategy(pred_ex, true_ex, val_ex)
                        total_acc1_ex.append(acc1_ex)
                        total_acc2_ex.append(acc2_ex)
                        total_acc3_ex.append(acc3_ex)
                        ex_count += true_ex.shape[0]
                        #total_profit_min_max += np.array(profit_min_max)
                        #total_profit_mean += np.array(profit_mean)
                        strategy_data = pd.concat([strategy_data, tmp_out])

                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                loss_local = abs(pred.detach().cpu().numpy() - true.detach().cpu().numpy())
                total_loss.append(loss)
                total_loss_local.append(loss_local)
                acc1, acc2, acc3, _ = _check_strategy(pred, true, val)
                total_acc1.append(acc1)
                total_acc2.append(acc2)
                total_acc3.append(acc3)

        total_loss = np.average(total_loss)
        total_loss_ex = np.average(total_loss_ex)
        total_loss_local = np.average(total_loss_local) * 10000000
        total_acc1 = np.average(total_acc1)
        total_acc1_ex = np.average(total_acc1_ex)
        total_acc2 = np.average(total_acc2)
        total_acc2_ex = np.average(total_acc2_ex)
        total_acc3 = np.average(total_acc3)
        total_acc3_ex = np.average(total_acc3_ex)
        trade_data = strategy_data.groupby('date').mean().reset_index()
        backtest_min_max, total_profit_min_max = _back_test_spot_swing(trade_data, threshold=15000,
                                                                       pred_opsion='min_max')
        backtest_mean, total_profit_mean = _back_test_spot_swing(trade_data, threshold=15000,
                                                                       pred_opsion='mean')

        self.model.train()
        ex_count = backtest_mean.shape[0]
        backtest_min_max.to_csv('backtest_min_max.csv')
        backtest_mean.to_csv('backtest_mean.csv')
        return total_loss, total_loss_local, total_acc1, total_acc2, total_acc3, total_loss_ex, total_acc1_ex, total_acc2_ex, total_acc3_ex, ex_count, total_profit_min_max, total_profit_mean

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
                        if self.args['extra'] == True:
                            pred_ex = pred[masks]
                            true_ex = true[masks]
                            if true_ex.shape[0] > 0:
                                loss_ex = criterion(pred_ex, true_ex)
                                loss_all = loss.item() + loss_ex.item()
                                train_loss.append(loss_all)
                            else:
                                loss_ex = None
                                train_loss.append(loss.item())

                        else:
                            train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        if (self.args['extra'] == True)&(loss_ex is not None):
                            loss += loss_ex
                            loss.backward()
                        else:
                            loss.backward()

                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_loss_local, acc1, acc2, acc3, vali_loss_ex, acc1_ex, acc2_ex, acc3_ex, ex_count, profit_min_max, profit_mean = self.vali(vali_data, vali_loader, criterion)

            mlflow.log_metric("Cost time", int(time.time()-epoch_time), step=epoch + 1)
            mlflow.log_metric("Train Loss", train_loss, step=epoch + 1)
            mlflow.log_metric("Vali Loss", vali_loss, step=epoch + 1)
            mlflow.log_metric("Vali Loss ex", vali_loss_ex, step=epoch + 1)
            mlflow.log_metric("ACC1", acc1, step=epoch + 1)
            mlflow.log_metric("ACC1 ex", acc1_ex, step=epoch + 1)
            mlflow.log_metric("ACC2", acc2, step=epoch + 1)
            mlflow.log_metric("ACC3", acc3, step=epoch + 1)
            mlflow.log_metric("EX count", ex_count, step=epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ACC1: {4:.7f} ACC2: {5:.7f} ACC3: {6:.7f}  Vali Loss ex: {7:.7f} ACC1 ex: {8:.7f} ACC2 ex: {9:.7f} ACC3 ex: {10:.7f} ex count: {11:.1f} Profit min max: {12} Profit mean: {13}".format(
                epoch + 1, train_steps, train_loss, vali_loss, acc1, acc2, acc3, vali_loss_ex, acc1_ex, acc2_ex, acc3_ex, ex_count, str(profit_min_max), str(profit_mean)))

            if acc1 > 0.6:
                torch.save(self.model.to('cpu').state_dict(), str(acc1)+'_best_model_checkpoint_cpu.pth')
            early_stopping(vali_loss, self.model, path)
            self.model.to(self.device)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model


    def predict(self, load=True):
        def _check_mergin(raw):
            output = [raw['date'].max()]
            max_target = raw['hi'].max()
            min_target = raw['lo'].min()
            min_pred = raw['pred'].min()
            max_pred = raw['pred'].max()
            spread_4 = (max_pred - min_pred) / 4
            spread_3 = (max_pred - min_pred) / 3
            fix_values = [0, spread_4, spread_3]

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

            out_sp = max_pred - min_pred
            max_check = max_pred < max_target
            min_check = min_pred>min_target
            return output , [raw['date'].max(), max_target, min_target, max_pred, min_pred, out_sp, max_check, min_check]
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
                #zip(seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, seq_raw):

            seq_x = torch.tensor(np.expand_dims(seq_x, axis=0))
            seq_y = torch.tensor(np.expand_dims(seq_y, axis=0))
            seq_x_mark = torch.tensor(np.expand_dims(seq_x_mark, axis=0))
            seq_y_mark = torch.tensor(np.expand_dims(seq_y_mark, axis=0))

            if seq_y.shape[1] == (self.args.label_len + self.args.pred_len):
                pred, _, _ = self._process_one_batch(
                    eval_data, seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y)

                pred_hi = pred[0, :, 0].detach().cpu().numpy()
                pred_lo = pred[0, :, 1].detach().cpu().numpy()
                raw = pd.DataFrame(raw[-self.args.pred_len:, :6], columns=cols)
                raw['pred_hi'] = pred_hi * 10000000
                raw['pred_lo'] = pred_lo * 10000000
                raw['pred'] = (raw['pred_hi'] + raw['pred_lo'])/2
                output = pd.concat([output, raw])
                out1, out2 = _check_mergin(raw)
                spread_out1.append(out1)
                spread_out2.append(out2)

        return output.reset_index(), pd.DataFrame(spread_out1), pd.DataFrame(spread_out2, columns=['date', 't_max', 't_min', 'p_max', 'p_min', 'spread', 'max_tf', 'min_tf'])

    def _create_masks(self, batch_y, mergin=30000):
        masks = []
        for hi_lo in batch_y:
            hi_max = hi_lo[: ,0].max()
            lo_min = hi_lo[: ,1].min()
            spread = (hi_max - lo_min) * 10000000
            if spread >= mergin:
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

            masks = self._create_masks(outputs)

            return outputs, batch_y, masks, batch_val

        except:
            print("-------------------prediction error-------------------")
            print(batch_x.shape)
            print(batch_x_mark.shape)
            print(dec_inp.shape)
            print(batch_y_mark.shape)
            print("------------------------------------------------------")
            return None, None
