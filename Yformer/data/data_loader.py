import os
import copy
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
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

class Dataset_BTC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='ALL', data_path='GMO_BTC_JPY_ohclv.csv',use_decoder_tokens=False,
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
        self.use_decoder_tokens = use_decoder_tokens

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
            df_raw = df_raw.dropna(how='any')
        elif self.option == 'feature_engineering':
            df_raw['spread'] = (df_raw['hi'] - df_raw['lo']) + 10
            df_raw = add_features_v2(df_raw, self.feature_add)[(self.feature_add - 1):]

            df_raw['transition'] = df_raw['cl'] - df_raw['op']
            df_raw['volatility'] = df_raw['spread'] - abs(df_raw['transition'])

        df_raw = df_raw.reset_index(drop=True)
        range1 = 0
        range2 = 6170
        range3 = 7500

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
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

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
        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
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

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='GMO_BTC_JPY_ohclv_eval.csv', use_decoder_tokens=False,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.use_decoder_tokens = use_decoder_tokens
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print("df_raw.shape :", df_raw.shape)
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            print(df_data.head())
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        print("data.shape :", data.shape)
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            print("df_stamp.shape : ", df_stamp.shape)
            data_stamp = df_stamp.drop(['date'],1).values
            print("df_stamp.shape after removing time stamp 1 : ", df_stamp.shape)

        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        print("data_x :", self.data_x.shape)
        self.data_y = data[border1:border2]
        print("data_y :", self.data_y.shape)

        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len 
            r_end = r_begin + self.label_len + self.pred_len


        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', use_decoder_tokens=False,
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_decoder_tokens = use_decoder_tokens
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x:x//15)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len


        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ECL_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='S', data_path='ECL.txt', use_decoder_tokens=False,
                target='320', scale=True, timeenc=0, freq='h', sample_frac=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.use_decoder_tokens = use_decoder_tokens
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path, header=None) # there are no headers
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        ftrs = []
        for i in range((df_raw.shape[1] - 2)):
            ftrs.append('ftr' + str(i))
        header = list(['date'] + ftrs + [self.target])
        df_raw.columns = header

        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.6818)
        num_test = int(len(df_raw)*0.1818)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
features='S', data_path='ECL.txt', use_decoder_tokens=False,
target='OT', scale=True, timeenc=0, freq='h', sample_frac=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.use_decoder_tokens = use_decoder_tokens
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        ftrs = []
        for i in range((df_raw.shape[1] - 2)):
            ftrs.append('ftr' + str(i))
        header = list(['date'] + ftrs + [self.target])
        df_raw.columns = header

        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.6818)
        num_test = int(len(df_raw)*0.1818)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, timeenc=0, freq='15min'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq[-1:])
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)