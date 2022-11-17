import os
import shutil
import time
from tqdm import tqdm
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings('ignore')
from data_process.etth_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
    Dataset_BTC, Dataset_BTC2, Dataset_BTC_pred
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decompose

from strategy import Estimation


class Exp_ETTh(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETTh, self).__init__(args)
        self.device = self._acquire_device()


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = 7
        elif self.args.features == 'MS':
            in_dim = 8
        else:
            print('Error!')

        if self.args.decompose:
            model = SCINet_decompose(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=in_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN)
        else:
            model = SCINet(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=in_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN)
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'BTC': Dataset_BTC,
            'BTC2': Dataset_BTC2,
            'BTC_pred': Dataset_BTC_pred,
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'val':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_BTC_pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        if self.args.data == 'BTC2':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                date_period1=args.date_period1,
                date_period2=args.date_period2,
                date_period3=args.date_period3,
                option= args.option
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        estimation = Estimation(self.args)

        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale)
                true_scales.append(true_scale)
                # Strategyモジュール追加
                batch_eval = batch_eval[:, -self.args.pred_len:, :]
                masks = self._create_masks(pred_scales, batch_eval)
                estimation.run_batch(index, pred_scale, true_scale, masks, batch_eval)

            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(),
                                                                                       true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale)
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale)
                # Strategyモジュール追加
                batch_eval = batch_eval[:, -self.args.pred_len:, :]
                masks = self._create_masks(pred_scales, batch_eval)
                estimation.run_batch(index, pred_scale, true_scale, masks, batch_eval)

            else:
                print('Error!')

            total_loss.append(loss)
        total_loss = np.average(total_loss)

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print(
                'denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes,
                                                                                                             rmses,
                                                                                                             mapes,
                                                                                                             mspes,
                                                                                                             corrs))


        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print(
                'mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse,
                                                                                                                   mae,
                                                                                                                   rmse,
                                                                                                                   mape,
                                                                                                                   mspe,
                                                                                                                   corr))
            print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
                mses, maes, rmses, mapes, mspes, corrs))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
                mse, mae, rmse, mape, mspe, corr))
            print(
                'final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
                    mses, maes, rmses, mapes, mspes, corrs))
        else:
            print('Error!')

        return total_loss, estimation

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter('event/run_ETTh/{}'.format(self.args.model_name))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data,
                                                     horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                if self.args.stacks == 1:
                    loss = criterion(pred, true)
                elif self.args.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())

                if self.args.use_amp:
                    print('use amp')
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            target_time_range_from = train_data.target_time_range_from
            target_time_range_to = train_data.target_time_range_to

            print('--------start to validate from {} to {}-----------'.format(target_time_range_from, target_time_range_to))
            valid_loss, estimation = self.valid(valid_data, valid_loader, criterion)
            acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, acc4_ex, cnt12, values12, dict12, strategy_data = estimation.run(epoch, target_time_range_from, target_time_range_to)

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            early_stopping(valid_loss, self.model, path)

            if epoch > 10:
                #score = values11[4] + values21[4]
                score = values12[0]
                best_model_path = '/'+setting+'_'+str(score)+'_'+str(epoch+1)+'_best/'
                if (score > 0.8):
                    if not os.path.exists(best_model_path):
                        save_path = self.args.save_path
                        os.makedirs(best_model_path)
                        strategy_data.to_csv(best_model_path + 'strategy_data.csv')
                        torch.save(self.model.to('cpu').state_dict(), best_model_path + 'checkpoint_cpu.pth')
                        pickle.dump(train_data.scaler, open(best_model_path + 'scaler.pkl', 'wb'))
                        pickle.dump(train_data.scaler_target, open(best_model_path + 'scaler_target.pkl', 'wb'))
                        new_path = shutil.move(best_model_path, save_path)
                        print(new_path)

            else:
                pass
            self.model.to(self.device)
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

        estimation.run(100)

        return loss

    def predict(self, mode, load=True):
        import pandas as pd
        timeenc = 0 if self.args.embed != 'timeF' else 1
        pred_data = Dataset_BTC_pred(
                    root_path=self.args.root_path,
                    data_path=self.args.data_path,
                    size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                    features=self.args.features,
                    target=self.args.target,
                    timeenc=timeenc,
                    freq=self.args.freq,
                    date_period1=self.args.date_period1,
                    date_period2=self.args.date_period2,
                    mode = mode
            )

        data_values, target_val, data_stamp, df_raw = pred_data.read_data()
        road_data = pred_data.extract_data(data_values, target_val, data_stamp, df_raw)

        if load:
            best_model_path = self.args.model_path + '/' + 'checkpoint_cpu.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        cols = ['date', 'op', 'hi', 'lo', 'cl']
        output = pd.DataFrame()
        preds = []
        for index, seq_x, seq_y, raw in tqdm(road_data):
            input_x = np.expand_dims(seq_x, 0)
            input_y = np.expand_dims(seq_y, 0)
            pred, true = self._pred_one_batch_SCINet(pred_data, input_x, input_y)
            raw_data = raw[-self.args.pred_len:, :]
            tmp_out = pd.DataFrame(raw_data, columns=cols)
            if mode == 1:
                tmp_out['pred'] = pred[0]
                output = pd.concat([output, tmp_out])
            else:
                pre = pred[0]
                target_index = index % (self.args.pred_len/2)
                from_index = int((self.args.pred_len / 2) - target_index-1)
                to_index = int(self.args.pred_len - target_index-1)
                preds.append(pre[from_index:to_index])
                if len(preds) == self.args.pred_len / 2:
                    preds = np.array(preds)
                    tmp_out = tmp_out[:int(self.args.pred_len / 2)]
                    tmp_out['pred'] = preds.mean(axis=0)[:, 0]
                    output = pd.concat([output, tmp_out])
                    preds = []


        return output.reset_index(drop=True)


    def _inverse_transform_batch(self, batch_values, scaler):
        output = []
        for values in batch_values:
            out = scaler.inverse_transform(values)
            output.append(out)
        return np.array(output)

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        if self.args.use_gpu:
            batch_x = batch_x.double().cuda()
        else:
            batch_x = batch_x.double()
        batch_y = batch_y.double()

        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')

        # if self.args.inverse:
        # outputs_scaled = dataset_object.inverse_transform(outputs)
        scaler = dataset_object.scaler_target
        outputs_scaled = self._inverse_transform_batch(outputs.detach().cpu().numpy(), scaler)
        if self.args.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features == 'MS' else 0
        if self.args.use_gpu:
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        else:
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        # batch_y_scaled = dataset_object.inverse_transform(batch_y)
        batch_y_scaled = self._inverse_transform_batch(batch_y.detach().cpu().numpy(), scaler)

        if self.args.stacks == 1:
            return outputs, outputs_scaled, 0, 0, batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')


    def _pred_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        batch_x = torch.tensor(batch_x).double()
        batch_y = torch.tensor(batch_y).double()

        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')


        scaler = dataset_object.scaler_target
        outputs_scaled = self._inverse_transform_batch(outputs.detach().cpu().numpy(), scaler)
        if self.args.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        batch_y_scaled = self._inverse_transform_batch(batch_y.detach().cpu().numpy(), scaler)

        if self.args.stacks == 1:
            return outputs_scaled, batch_y_scaled
        elif self.args.stacks == 2:
            #return outputs_scaled, mid_scaled, batch_y_scaled
            return outputs_scaled, batch_y_scaled
        else:
            print('Error!')

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