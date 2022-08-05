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

        if flag == 'val':
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

    def vali(self, epoch ,vali_data, vali_loader, criterion):
        def _create_tmp_data(pred_data, true_data, val_data, index, option='mean'):
            output = pd.DataFrame()
            index = index.detach().cpu().numpy()
            for i, true, pred, val in zip(index, true_data, pred_data, val_data):
                true = true.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                val = val.detach().cpu().numpy()

                if option == 'mean':
                    target_index = i % 6
                    from_index = 6 - target_index - 1
                    to_index = 12 - target_index - 1
                    true = true[from_index:to_index]
                    pred = pred[from_index:to_index]
                    val = val[from_index:to_index]

                tmp_values = np.concatenate([val, true, pred], axis=1) * 10000000
                columns = ['date', 'op', 'cl', 'hi', 'lo', 'true', 'pred']

                tmp_data = pd.DataFrame(tmp_values, columns=columns)
                output = pd.concat([output, tmp_data])

            return output

        def _back_test_mm(trade_data, threshold=10000, num=12):
            def drop_off_sell_stocks(stocks, lo):
                output = []
                for i, stock in enumerate(stocks):
                    date = stock[0]
                    stock_price = stock[1]
                    stay_count = stock[2]
                    drop_off = stock[3]
                    diff = stock[4]
                    if drop_off == False:
                        if stock_price > lo:
                            diff = stock_price - lo
                            drop_off = True
                        else:
                            stay_count += 1
                        output.append([date, stock_price, stay_count, drop_off, diff])
                    else:
                        output.append(stock)

                return output

            def drop_off_buy_stocks(stocks, hi):
                output = []
                for i, stock in enumerate(stocks):
                    date = stock[0]
                    stock_price = stock[1]
                    stay_count = stock[2]
                    drop_off = stock[3]
                    diff = stock[4]
                    if drop_off == False:
                        if stock_price < hi:
                            diff = hi - stock_price
                            drop_off = True
                        else:
                            stay_count += 1
                        output.append([date, stock_price, stay_count, drop_off, diff])
                    else:
                        output.append(stock)

                return output

            def calc_stock_count(buy_stocks, sell_stocks):
                stock_count = 0
                for b_stock in buy_stocks:
                    if b_stock[3] == False:
                        stock_count += 1

                for s_stock in sell_stocks:
                    if s_stock[3] == False:
                        stock_count += 1

                return stock_count
            output = []
            buy_stocks = []
            sell_stocks = []
            stock_counts = []
            total = 0
            trade_cnt = 0
            max_stocks = 0
            for i in range(0, trade_data.shape[0], num):
                if i == 0:
                    start = 0
                    end = num - 1
                else:
                    end = i + num - 1

                tmp_data = trade_data.loc[start:end]
                pred_spread_min = int(tmp_data['pred'].min())
                pred_spread_max = int(tmp_data['pred'].max())

                spread_mergin = (pred_spread_max - pred_spread_min)

                buy = False
                sell = False
                trade = False

                for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                    buy_stocks = drop_off_buy_stocks(buy_stocks, hi)
                    sell_stocks = drop_off_sell_stocks(sell_stocks, lo)

                    if spread_mergin >= threshold:
                        trade = True
                        if hi > pred_spread_max:
                            sell = True
                            sell_price = pred_spread_max
                            sell_date = date
                        if lo < pred_spread_min:
                            buy = True
                            buy_price = pred_spread_min
                            buy_date = date

                stocks_count = calc_stock_count(buy_stocks, sell_stocks)
                stock_counts.append(stocks_count)
                if stocks_count > max_stocks:
                    max_stocks = stocks_count

                close_date = tmp_data['date'].values[-1]
                profit = 0

                if trade == True:
                    if (sell == True) & (buy == True):
                        profit = sell_price - buy_price
                    elif (sell == True) & (buy == False):
                        sell_stocks.append([sell_date, sell_price, 0, False, 0])
                    elif (sell == False) & (buy == True):
                        buy_stocks.append([buy_date, buy_price, 0, False, 0])
                    else:
                        pass

                    total += profit
                    trade_cnt += 1
                else:
                    pass

                output.append([close_date, total, profit, buy, sell])
                start = end + 1

            output = pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell'])
            output = output[output['profit'] != 0]
            term = (output['buy'] == True) & (output['sell'] == True)
            profit_win = output.loc[term, 'profit'].sum()

            stock_mean = np.mean(stock_counts)

            if stock_mean < 1:
                stock_mean = 1
            total = np.round(total / stock_mean, 2) / 1000000
            profit_win = np.round(profit_win / stock_mean, 2) / 1000000

            return output, (total, profit_win, stock_mean, max_stocks)

        def _back_test_spot_swing(trade_data, threshold=15000, num=12):
            output = []
            total = 0
            trade_cnt = 0
            for i in range(0, trade_data.shape[0], num):
                if i == 0:
                    start = 0
                    end = num - 1
                else:
                    end = i + num - 1

                tmp_data = trade_data.loc[start:end]
                base_price = tmp_data['op'].values[0]
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

            total = np.round(total / 1000000, 2)
            profit_win = np.round(profit_win / 1000000, 2)
            profit_loss = np.round(profit_loss / 1000000, 2)

            return output, (total, profit_win, profit_loss)

        def _check_strategy(pred_data, true, val, eval_masks, index):
            if eval_masks is not None:
                tmp_out1 = _create_tmp_data(pred_data[eval_masks], true[eval_masks], val[eval_masks], index, option='none')
                tmp_out2 = _create_tmp_data(pred_data, true, val, index, option='mean')
            else:
                tmp_out1 = []
                tmp_out2 = []

            pred = pred_data[:,:,0].detach().cpu() * 10000000
            true = true[:,:,0].detach().cpu() * 10000000
            val = val.detach().cpu() * 10000000

            pred_min = pred.min(axis=1)[0]
            #true_min = true.min(axis=1)[0]
            true_min = val[:, :, 4].min(axis=1)[0]
            pred_max = pred.max(axis=1)[0]
            #true_max = true.max(axis=1)[0]
            true_max = val[:, :, 3].max(axis=1)[0]
            op = val[:,0,1]

            spread_4 = (pred_max - pred_min) / 4
            spread_3 = (pred_max - pred_min) / 3

            acc1 = (pred_min > true_min) & (pred_max < true_max) & ((pred_max - pred_min) > 0)
            acc2 = ((pred_min + spread_4) > true_min) & ((pred_max - spread_4) < true_max) & ((pred_max - pred_min) > 0)

            acc3 = []
            for p_min, p_max, t_min, t_max, base in zip(pred_min, pred_max, true_min, true_max, op):
                spread1 = p_max - base
                spread2 = base - p_min

                if spread1 > spread2:
                    if t_max > p_max:
                        out = True
                    else:
                        out = False
                else:
                    if t_min < p_min:
                        out = True
                    else:
                        out = False
                acc3.append(out)

            acc3 = torch.tensor(acc3)

            return acc1.sum()/pred.shape[0], acc2.sum()/pred.shape[0], acc3.sum()/pred.shape[0], tmp_out1, tmp_out2

        def execute_back_test(backtest, input_dict):
            trade_data = input_dict['trade_data']
            num = input_dict['num']
            thresh_list = input_dict['thresh_list']
            best_score = 0
            for i, thresh in enumerate(thresh_list):
                threshold = thresh
                output, scores = backtest(trade_data, threshold=threshold, num=num)
                if (scores[0] > best_score) or (i == 0):
                    best_score = scores[0]
                    best_output = output
                    best_score_values = scores
                    out_dict = {'vesion': num, 'thresh': threshold}

            return best_output, best_score_values, out_dict

        self.model.eval()
        total_loss = []
        total_loss_ex = []
        total_loss_local = []
        total_acc1 = []
        total_acc2 = []
        total_acc3 = []
        total_acc1_ex = []
        total_acc2_ex = []
        total_acc3_ex = []
        strategy_data1 = pd.DataFrame()
        strategy_data2 = pd.DataFrame()
        for i, (index,batch_x,batch_y,batch_x_mark,batch_y_mark,batch_val) in enumerate(vali_loader):
            if (batch_y.shape[1] == (self.args.label_len + self.args.pred_len)) & \
                    (batch_x.shape[1] == self.args.seq_len):
                eval_masks = index % 12 == 0
                pred, true, masks, val = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_val)

                if self.args['extra'] == True:
                    pred_ex = pred[masks]
                    true_ex = true[masks]
                    val_ex = val[masks]
                    if true_ex.shape[0] > 0:
                        loss_ex = criterion(pred_ex.detach().cpu(), true_ex.detach().cpu())
                        total_loss_ex.append(loss_ex)
                        acc1_ex, acc2_ex, acc3_ex, _, _ = _check_strategy(pred_ex, true_ex, val_ex, None, None)
                        total_acc1_ex.append(acc1_ex)
                        total_acc2_ex.append(acc2_ex)
                        total_acc3_ex.append(acc3_ex)


                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                loss_local = abs(pred.detach().cpu().numpy() - true.detach().cpu().numpy())
                total_loss.append(loss)
                total_loss_local.append(np.average(loss_local))
                acc1, acc2, acc3, tmp_out1, tmp_out2 = _check_strategy(pred, true, val, eval_masks, index)
                total_acc1.append(acc1)
                total_acc2.append(acc2)
                total_acc3.append(acc3)
                strategy_data1 = pd.concat([strategy_data1, tmp_out1])
                strategy_data2 = pd.concat([strategy_data2, tmp_out2])

        tl = np.average(total_loss)
        tl_ex = np.average(total_loss_ex)
        acc1 = np.average(total_acc1)
        acc2 = np.average(total_acc2)
        acc3 = np.average(total_acc3)
        acc1_ex = np.average(total_acc1_ex)
        acc2_ex = np.average(total_acc2_ex)
        acc3_ex = np.average(total_acc3_ex)
        strategy_data1 = strategy_data1.reset_index(drop=True)
        strategy_data2 = strategy_data2.groupby('date').mean().reset_index()

        if epoch + 1 >= 10:
            input_dict1 = {'trade_data': strategy_data1, 'num': 12, 'thresh_list': [10000, 15000, 20000, 25000]}
            best_output11, values11, dict11 = execute_back_test(_back_test_spot_swing, input_dict1)
            best_output12, values12, dict12 = execute_back_test(_back_test_mm, input_dict1)

            input_dict2 = {'trade_data': strategy_data2, 'num': 6, 'thresh_list': [5000, 10000, 15000, 20000]}
            best_output21, values21, dict21 = execute_back_test(_back_test_spot_swing, input_dict2)
            best_output22, values22, dict22 = execute_back_test(_back_test_mm, input_dict2)


            cnt11 = best_output11.shape[0]
            cnt21 = best_output21.shape[0]
            best_output11.to_csv('best_output11.csv')
            best_output21.to_csv('best_output21.csv')
            strategy_data1.to_csv('strategy_data1.csv')
            strategy_data2.to_csv('strategy_data2.csv')
        else:
            cnt11 = values11 = dict11 = cnt21 = values21 = dict21 = values12 = dict12 = values22 = dict22 = None

        self.model.train()
        return tl, tl_ex, acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, cnt11, values11, dict11, cnt21, values21, dict21, values12, dict12, values22, dict22

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
                        loss_ex = None
                        if self.args['extra'] == True:
                            pred_ex = pred[masks]
                            true_ex = true[masks]
                            if true_ex.shape[0] > 0:
                                loss_ex = criterion(pred_ex, true_ex)
                                loss_all = loss.item() + loss_ex.item()
                                train_loss.append(loss_all)
                            else:
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
            vali_loss, vali_loss_ex, acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, cnt11, values11, dict11, cnt21, values21, dict21, values12, dict12, values22, dict22 = self.vali(epoch, vali_data, vali_loader, criterion)

            mlflow.log_metric("Cost time", int(time.time()-epoch_time), step=epoch + 1)
            mlflow.log_metric("Train Loss", train_loss, step=epoch + 1)
            mlflow.log_metric("Vali Loss", vali_loss, step=epoch + 1)
            mlflow.log_metric("Vali Loss ex", vali_loss_ex, step=epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali Loss ex: {4:.7f} ACC1: {5:.5f} ACC2: {6:.5f} ACC3: {7:.5f}  ACC1Ex: {8:.5f} ACC2Ex: {9:.5f} ACC3Ex: {10:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_loss_ex, acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex))

            if epoch+1 >= 10:
                print("Test1 | Swing - cnt: {0} best profit: {1} config: {2}  MM bot - best profit: {3} config: {4}".format(
                    cnt11, values11, dict11, values12, dict12))
                print("Test2 | Swing - cnt: {0} best profit: {1} config: {2}  MM bot - best profit: {3} config: {4}".format(
                    cnt21, values21, dict21, values22, dict22))

            early_stopping(-acc1, self.model, path)
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
                raw['pred_hi'] = pred_hi * 10000000
                raw['pred_lo'] = pred_lo * 10000000
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
            spread1 = (hi_max - op) * 10000000
            spread2 = (op - lo_min) * 10000000
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
