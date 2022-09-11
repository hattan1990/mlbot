import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import copy

from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


def add_features(data, num):
    columns = ['op', 'hi', 'lo', 'cl', 'volume']
    for i in range(1, num):
        for col in columns:
            data[col + '_' + str(i)] = data[col].pct_change(periods=i)
    return data

def add_features_v2(data, num):
    columns = ['op', 'hi', 'lo', 'cl', 'volume', 'spread']
    for i in range(1, num):
        for col in columns:
            data[col + '_' + str(i)] = data[col].pct_change(periods=i)
    return data

def add_ext_features(data, num=2):
    columns = data.columns[1:]
    for i in range(1, num):
        for col in columns:
            data[col + '_' + str(i)] = data[col].pct_change(periods=i)
    data = data.drop(columns=columns)
    return data

class EvalDataset():
    def __init__(self, root_path, size=[48, 24, 12],
                 features='ALL', data_path='GMO_BTC_JPY_ohclv_eval.csv',
                 target='cl', inverse=False, timeenc=1, freq='t', feature_add=0,
                 option='pct'):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.option = option
        self.feature_add = feature_add
        if self.option == 'pct':
            self.scaler = pickle.load(open('./weights/scaler.pkl', 'rb'))
        else:
            self.scaler = pickle.load(open('./weights/scaler_add.pkl', 'rb'))

    def read_data(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop(columns="Unnamed: 0")
        df_raw = df_raw.fillna(method='ffill')
        df_raw = df_raw.fillna(method='bfill')

        if self.option == 'pct':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add-1):]
        elif self.option == 'mean':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add - 1):]
            num = 24
            df_raw['hi_mean'] = df_raw['hi'].rolling(num).mean()
            df_raw['lo_mean'] = df_raw['lo'].rolling(num).mean()
            df_raw = df_raw.dropna(how='any')
        elif self.option == 'feature_engineering':
            df_raw['spread'] = (df_raw['hi'] - df_raw['lo']) + 10
            df_raw = add_features_v2(df_raw, self.feature_add)[(self.feature_add - 1):]

            df_raw['transition'] = df_raw['cl'] - df_raw['op']
            df_raw['volatility'] = df_raw['spread'] - abs(df_raw['transition'])

        df_raw = df_raw.reset_index(drop=True)
        data = copy.deepcopy(df_raw)
        cols_data = data.columns[1:]
        df_data = data[cols_data]
        data_values = self.scaler.transform(df_data.values)
        data_values = data_values[:, 5:]

        if self.option == 'mean':
            data_values = data_values[:, :-2]


        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.target == None:
            target_val = data_values
        else:
            target_val = (df_data[self.target[0]] + df_data[self.target[1]]) / 2
            target_val = target_val.values / 10000000
            target_val = np.expand_dims(target_val, 1)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        return data_values, target_val, data_stamp, df_raw

    def extract_data(self, data_values, target_val, data_stamp, df_raw):
        data_len = len(data_values) - self.seq_len
        for index in range(0, data_len, self.pred_len):
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = data_values[s_begin:s_end]
            seq_y = target_val[r_begin:r_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]
            seq_raw = df_raw.values[r_begin:r_end]
            yield seq_x, seq_y, seq_x_mark, seq_y_mark, seq_raw

class Dataset_BTC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='ALL', data_path='GMO_BTC_JPY_ohclv.csv',
                 target='cl', scale=True, inverse=False, timeenc=0, freq='t',
                 feature_add=0, option='pct', eval_mode=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]


        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.option = option
        self.feature_add = feature_add
        self.eval_mode = eval_mode
        self.__read_data__()

    def __read_data__(self):
        if self.set_type == 2:
            self.scaler = pickle.load(open('scaler.pkl', 'rb'))
        else:
            self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop(columns="Unnamed: 0")

        df_raw = df_raw.fillna(method='ffill')
        df_raw = df_raw.fillna(method='bfill')

        if self.option == 'pct':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add-1):]
        elif self.option == 'mean':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add - 1):]
            num = 12
            df_raw['hi_mean'] = df_raw['hi'].rolling(num).mean()
            df_raw['lo_mean'] = df_raw['lo'].rolling(num).mean()
            #df_raw['hi_mean'] = np.append(df_raw['hi_mean'].values[num - 1:], np.array([np.nan] * (num - 1)))
            #df_raw['lo_mean'] = np.append(df_raw['lo_mean'].values[num - 1:], np.array([np.nan] * (num - 1)))
            df_raw = df_raw.dropna(how='any')
        elif self.option == 'feature_engineering':
            df_raw['spread'] = (df_raw['hi'] - df_raw['lo']) + 10
            df_raw = add_features_v2(df_raw, self.feature_add)[(self.feature_add - 1):]

            df_raw['transition'] = df_raw['cl'] - df_raw['op']
            df_raw['volatility'] = df_raw['spread'] - abs(df_raw['transition'])

        df_raw = df_raw.reset_index(drop=True)
        range1 = 0
        range2 = 617000
        range3 = 750000

        if self.data_path == 'GMO_BTC_JPY_ohclv5.csv':
            range2 = int(range2 / 5)
            range3 = int(range3 / 5)
        else:
            pass

        if self.set_type == 0:
            df_raw = df_raw[df_raw['date'] >= '2020-12-02 00:00']
        elif self.set_type == 1:
            df_raw = df_raw[df_raw['date'] >= '2022-03-01 00:00:00']

        border1s = [range1, range1, range1]
        border2s = [range2, range3, range3]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'ALL':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
            pickle.dump(self.scaler, open("scaler.pkl", "wb"))

        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2, 5:]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            if self.target == None:
                self.data_y = data[border1:border2]
            else:
                if self.option == 'mean':
                    hi_lo = (df_data[self.target[0]+'_mean'] + df_data[self.target[1]+'_mean']) / 2
                    hi_lo = hi_lo.values[border1:border2] / 10000000
                    self.data_y = np.expand_dims(hi_lo, 1)
                    self.data_x = self.data_x[:, :-2]
                else:
                    hi_lo = (df_data[self.target[0]] + df_data[self.target[1]]) / 2
                    hi_lo = hi_lo.values[border1:border2] / 10000000
                    self.data_y = np.expand_dims(hi_lo, 1)
        self.data_stamp = data_stamp
        df_raw['date'] = df_raw['date'].apply(lambda x:int(x[:4]+x[5:7]+x[8:10]+x[11:13]+x[14:16]))
        self.data_val = df_raw[['date', 'op', 'cl', 'hi', 'lo']].values[border1:border2] / 10000000

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_val = self.data_val[r_begin:r_end]
        if self.eval_mode:
            return index, seq_x, seq_y, seq_x_mark, seq_y_mark, seq_val
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_val

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
