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

class EvalDataset():
    def __init__(self, root_path, size=[48, 24, 12],
                 features='ALL', data_path='gmo_btcjpy_ohlcv_val.csv',
                 target='cl', inverse=False, timeenc=1, freq='t', feature_add=0):

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

        self.option = 'pct'
        self.feature_add = feature_add
        self.scaler = pickle.load(open('scaler.pkl', 'rb'))

    def read_data(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        d1 = pd.read_csv(os.path.join(self.root_path,
                                      'nasdaq_mega_ohlcv_daily.csv'))
        d2 = pd.read_csv(os.path.join(self.root_path,
                                      'us_markets_ohlcv_daily.csv'))
        ext_data = pd.merge(d1, d2, on='index', how='left')
        ext_data = ext_data.fillna(method='ffill')

        df_raw['index'] = df_raw['date'].apply(lambda x: x[:10])
        df_raw = pd.merge(df_raw, ext_data, on='index', how='left')
        df_raw = df_raw.fillna(method='ffill')
        df_raw = df_raw.fillna(method='bfill')
        df_raw = df_raw.drop(columns='index')

        if self.option == 'pct':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add-1):]

        df_raw = df_raw.reset_index(drop=True)
        data = copy.deepcopy(df_raw)
        cols_data = data.columns[1:]
        df_data = data[cols_data]
        data_values = self.scaler.transform(df_data.values)
        data_values = data_values[:, 5:]


        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.target == None:
            target_val = data_values
        else:
            #target_col1 = df_data.columns.to_list().index(self.target[0])
            #target_col2 = df_data.columns.to_list().index(self.target[1])
            #target_val = data_values[:, [target_col1, target_col2]]
            target_val = df_data[[self.target[0], self.target[1]]].values / 10000000
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        return data_values, target_val, data_stamp, df_raw

    def extract_data(self, data_values, target_val, data_stamp, df_raw):
        data_len = len(data_values) - self.seq_len
        seq_x = []
        seq_y = []
        seq_x_mark = []
        seq_y_mark = []
        seq_raw = []

        for index in range(0, data_len, self.pred_len):
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x.append(data_values[s_begin:s_end])
            seq_y.append(target_val[r_begin:r_end])
            seq_x_mark.append(data_stamp[s_begin:s_end])
            seq_y_mark.append(data_stamp[r_begin:r_end])
            seq_raw.append(df_raw.values[r_begin:r_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_raw

class Dataset_BTC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='ALL', data_path='gmo_btcjpy_ohlcv.pkl',
                 target='cl', scale=True, inverse=False, timeenc=0, freq='t', feature_add=0):
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

        self.option = 'pct'
        self.feature_add = feature_add
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        d1 = pd.read_csv(os.path.join(self.root_path,
                                          'nasdaq_mega_ohlcv_daily.csv'))
        d2 = pd.read_csv(os.path.join(self.root_path,
                                          'us_markets_ohlcv_daily.csv'))
        ext_data = pd.merge(d1, d2, on='index', how='left')
        ext_data = ext_data.fillna(method='ffill')

        df_raw['index'] = df_raw['date'].apply(lambda x: x[:10])
        df_raw = pd.merge(df_raw, ext_data, on='index', how='left')
        df_raw = df_raw.fillna(method='ffill')
        df_raw = df_raw.fillna(method='bfill')
        df_raw = df_raw.drop(columns='index')

        if self.option == 'pct':
            df_raw = add_features(df_raw, self.feature_add)[(self.feature_add-1):]

        df_raw = df_raw.reset_index(drop=True)

        border1s = [0, 12 * 30 * 1100 + 4 * 30 * 1100 - self.seq_len]
        border2s = [12 * 30 * 1100 + 4 * 30 * 1100, 12 * 30 * 1000 + 8 * 30 * 1000]
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
            #data = df_data.values
            #data[:,:4] = data[:,:4] / 10000000
            #data[:, 4] = data[:, 4] / 500
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
                #target_col1 = df_data.columns.to_list().index(self.target[0])
                #target_col2 = df_data.columns.to_list().index(self.target[1])
                self.data_y = df_data[[self.target[0], self.target[1]]].values[border1:border2] / 10000000
        self.data_stamp = data_stamp

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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
