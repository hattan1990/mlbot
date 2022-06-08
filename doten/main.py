import requests
import json
import pickle
import numpy as np
import pandas as pd
import time
from dateutil import parser
from datetime import datetime
from datetime import timedelta
import logging

from utils import GMOHandler

import warnings
warnings.simplefilter('ignore')


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    h1 = logging.StreamHandler()
    h1.setLevel(logging.DEBUG)
    h2 = logging.FileHandler('ERROR.log')
    h2.setLevel(logging.ERROR)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    h1.setFormatter(fmt)
    h2.setFormatter(fmt)

    logger.addHandler(h1)
    logger.addHandler(h2)
    return logger


class Trade:
    def __init__(self):
        self.model_long = pickle.load(open('./models/long_model.pkl', 'rb'))
        self.model_short = pickle.load(open('./models/short_model.pkl', 'rb'))
        self.logger = get_logger()
        self.flgs = self.init_action_flags({})


    def init_action_flags(self, flgs):
        flgs['long_buy'] = np.zeros(31)
        flgs['long_sell'] = np.zeros(31)
        flgs['short_buy'] = np.zeros(31)
        flgs['short_sell'] = np.zeros(31)
        return flgs

    def update_action_flgs(self, flg1, flg2):
        flg1 = np.append(flg1, 1)
        flg1 = np.delete(flg1, 0)
        flg2 = np.append(flg2, 0)
        flg2 = np.delete(flg2, 0)
        return flg1, flg2

    def update_no_action_flgs(self, flg1, flg2):
        flg1 = np.append(flg1, 0)
        flg1 = np.delete(flg1, 0)
        flg2 = np.append(flg2, 0)
        flg2 = np.delete(flg2, 0)
        return flg1, flg2

    def get_data(self):
        def daterange(_start, _end):
            for n in range((_end - _start).days):
                yield _start + timedelta(n)

        now = datetime.now()
        now_f = now.strftime('%Y-%m-%d')
        start = datetime.strptime(now_f, '%Y-%m-%d') - timedelta(4)
        end = datetime.strptime(now_f, '%Y-%m-%d') + timedelta(1)
        endPoint = 'https://api.coin.z.com/public'
        price_data = []

        for d in daterange(start, end):
            path = '/v1/klines?symbol=BTC_JPY&interval=1min&date={}'.format(d.strftime('%Y%m%d'))
            response_1 = requests.get(endPoint + path)
            try:
                for i in response_1.json()['data']:
                    price_data.append(
                        {"timestamp": datetime.fromtimestamp(int(i["openTime"]) / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                         "op": int(i["open"]),
                         "hi": int(i["high"]),
                         "lo": int(i["low"]),
                         "cl": int(i["close"]),
                         "volume": float(i["volume"])
                         })
            except:
                self.logger.error("Could not get the data:%s", d.strftime('%Y%m%d'))

        df = pd.DataFrame(price_data)

        return df

    def get_recursive_data(self, df, base_time):
        cnt = 1
        while (base_time.strftime("%Y-%m-%d %H:%M:%S") not in df.timestamp.values):
            time.sleep(0.5)
            self.logger.info("get_recursive_data | count %s", cnt)
            df = self.get_data()
            cnt += 1
            if cnt >= 10:
                break

        return df

    def get_target_timestamp(self):
        base_time = parser.parse(datetime.now().strftime("%Y%m%d %H:%M"))
        while True:
            time.sleep(1)
            now = datetime.now()
            now_f = parser.parse(now.strftime("%Y%m%d %H:%M"))
            if now_f > base_time:
                return base_time

    def add_features(self, dataset, span):
        try:
            columns = dataset.columns[1:]
            for i in range(1, span):
                for col in columns:
                    dataset[col + '_' + str(i)] = dataset[col].pct_change(periods=i)

            return dataset
        except:
            self.logger.error("error | add_features")
            return None

    def apply_trend(self, df):
        try:
            df['rolling_avg1day'] = df['cl'].rolling(60 * 24 * 1).mean()
            df['rolling_avg3day'] = df['cl'].rolling(60 * 24 * 3).mean()

            df['diff_avg3day'] = df['rolling_avg3day'].diff()

            up_avg3day = df['diff_avg3day'] >= 0
            down_avg3day = df['diff_avg3day'] < 0

            rise = df['rolling_avg1day'] > df['rolling_avg3day']
            fall = df['rolling_avg1day'] < df['rolling_avg3day']

            df.loc[up_avg3day & rise, 'trend'] = 'かなり上昇'
            df.loc[up_avg3day & fall, 'trend'] = '上昇'
            df.loc[down_avg3day & fall, 'trend'] = 'かなり下降'
            df.loc[down_avg3day & rise, 'trend'] = '下降'
            df['trend'] = df['trend'].fillna('-')

            return df
        except:
            self.logger.error("error | apply_trend")

            return None


    def inference_buy_sell(self, base_time, X, trend):
        long_buy_flgs = self.flgs['long_buy']
        long_sell_flgs = self.flgs['long_sell']
        short_buy_flgs = self.flgs['short_buy']
        short_sell_flgs = self.flgs['short_sell']
        model_long = self.model_long
        model_short = self.model_short
        position = None

        if trend in ("上昇", "かなり上昇"):
            pred = model_long.predict(X, num_iteration=model_long.best_iteration)
            action = pred.argmax(axis=1)[0]
            if action == 1:
                long_buy_flgs, long_sell_flgs = self.update_action_flgs(long_buy_flgs, long_sell_flgs)
            elif action == 2:
                long_sell_flgs, long_buy_flgs = self.update_action_flgs(long_sell_flgs, long_buy_flgs)
            else:
                long_buy_flgs, long_sell_flgs = self.update_no_action_flgs(long_buy_flgs, long_sell_flgs)

            short_buy_flgs = np.zeros(31)
            short_sell_flgs = np.zeros(31)
            position = 'long'

        elif trend in ("下降", "かなり下降"):
            pred = model_short.predict(X, num_iteration=model_short.best_iteration)
            action = pred.argmax(axis=1)[0]
            if action == 2:
                short_buy_flgs, short_sell_flgs = self.update_action_flgs(short_buy_flgs, short_sell_flgs)
            elif action == 1:
                short_sell_flgs, short_buy_flgs = self.update_action_flgs(short_sell_flgs, short_buy_flgs)
            else:
                short_buy_flgs, short_sell_flgs = self.update_no_action_flgs(short_buy_flgs, short_sell_flgs)

            long_buy_flgs = np.zeros(31)
            long_sell_flgs = np.zeros(31)
            position = 'short'

        else:
            self.logger.error("not mutch trend")

        self.flgs['long_buy'] = long_buy_flgs
        self.flgs['long_sell'] = long_sell_flgs
        self.flgs['short_buy'] = short_buy_flgs
        self.flgs['short_sell'] = short_sell_flgs
        self.logger.info("inference_buy_sell | %s trend:%s position:%s action:%s", base_time, trend, position, action)

        return self.flgs, position

    def order_position(self, buy_flg, sell_flg, position_flg):
        buy = False
        sell = False
        buy_signal = buy_flg.sum()
        sell_signal = sell_flg.sum()
        if position_flg == False:
            if buy_signal >= 15:
                buy = True
        elif position_flg == True:
            if sell_signal >= 15:
                sell = True
        else:
            self.logger.error("error order_position | {%s} buy_signal {%s} sell_signal {%s} ", position_flg, buy_signal,
                         sell_signal)

        return buy, sell


def main():
    trade = Trade()
    trade.logger.info("Start BTC trade")

    long_position_flg = False
    short_position_flg = False
    long_buy_price = None
    long_sell_price = None
    short_buy_price = None
    short_sell_price = None
    now = datetime.now()
    execution_term = (now.weekday() == 2) & (now.hour == 15) == False

    while True:
        if execution_term:
            # データ抽出
            base_time = trade.get_target_timestamp()
            time.sleep(1)
            df = trade.get_data()
            df = trade.get_recursive_data(df, base_time)

            # データの加工
            df = trade.add_features(df, 60)
            pred_columns = df.columns[6:]
            df = trade.apply_trend(df)


            base_time_str = base_time.strftime("%Y-%m-%d %H:%M:%S")
            pred_df = df[df['timestamp'] == base_time_str]
            if pred_df.shape[0] > 0:
                # モデルの予測
                trend = pred_df['trend'].values[0]
                X = pred_df[pred_columns].values
                flgs, position = trade.inference_buy_sell(base_time_str, X, trend)

                if position == 'long':
                    if short_position_flg == True:
                        print(position)
                    else:
                        print(position)

                elif position == 'short':
                    if long_position_flg == True:
                        print(position)
                    else:
                        print(position)
                else:
                    trade.logger.error("error manage_position")
            else:
                trade.logger.info("skip processing {%s}", base_time.strftime("%Y-%m-%d %H:%M:%S"))
                pass
        else:
            trade.logger.info("GMO server is under maintenance : %s", now.strftime("%Y-%m-%d %H:%M:%S"))
            pass
        now = datetime.now()
        execution_term = (now.weekday() == 2) & (now.hour == 15) == False
