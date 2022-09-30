import torch
import pandas as pd
import numpy as np
from dateutil import parser as ps
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Estimation:
    def __init__(self, args):
        self.args = args
        self.total_acc1 = []
        self.total_acc2 = []
        self.total_acc3 = []
        self.total_acc1_ex = []
        self.total_acc2_ex = []
        self.total_acc3_ex = []
        self.total_acc4_ex = []
        self.strategy_data1 = pd.DataFrame()
        self.strategy_data2 = pd.DataFrame()


    def check_strategy(self, pred_data, true, val, eval_masks, index):
        def _create_tmp_data(pred_data, true_data, val_data, index, option='mean'):
            output = pd.DataFrame()
            index = index.detach().cpu().numpy()
            for i, true, pred, val in zip(index, true_data, pred_data, val_data):
                true = true
                pred = pred

                if option == 'mean':
                    pred_len = self.args.pred_len
                    target_index = i % (pred_len/2)
                    from_index = int((pred_len/2) - target_index - 1)
                    to_index = int(pred_len - target_index - 1)
                    true = true[from_index:to_index]
                    pred = pred[from_index:to_index]
                    val = val[from_index:to_index]

                tmp_values = np.concatenate([val, true, pred], axis=1)
                columns = ['date', 'op', 'cl', 'hi', 'lo', 'true', 'pred']

                tmp_data = pd.DataFrame(tmp_values, columns=columns)
                output = pd.concat([output, tmp_data])

            return output

        if eval_masks is not None:
            tmp_out1 = _create_tmp_data(pred_data[eval_masks], true[eval_masks], val[eval_masks], index, option='none')
            tmp_out2 = _create_tmp_data(pred_data, true, val, index, option='mean')
        else:
            tmp_out1 = []
            tmp_out2 = []

        pred = pred_data[:, :, 0]
        true = true[:, :, 0]
        val = val.numpy()

        pred_min = pred.min(axis=1)
        # true_min = true.min(axis=1)[0]
        true_min = val[:, :, 4].min(axis=1)
        pred_max = pred.max(axis=1)
        # true_max = true.max(axis=1)[0]
        true_max = val[:, :, 3].max(axis=1)
        op = val[:, 0, 1]

        spread_4 = (pred_max - pred_min) / 4

        acc1 = (pred_min > true_min) & (pred_max < true_max) & ((pred_max - pred_min) > 0)
        acc2 = ((pred_min + spread_4) > true_min) & ((pred_max - spread_4) < true_max) & ((pred_max - pred_min) > 0)

        acc3 = []
        acc4 = []
        for p_min, p_max, t_min, t_max, base in zip(pred_min, pred_max, true_min, true_max, op):
            spread1 = p_max - base
            spread2 = base - p_min

            if spread1 > spread2:
                if t_max > p_max:
                    out1 = True
                    out2 = True
                elif t_max > p_max - (spread1 / 4):
                    out1 = False
                    out2 = True
                else:
                    out1 = False
                    out2 = False
            else:
                if t_min < p_min:
                    out1 = True
                    out2 = True
                elif t_min < p_min + (spread2 / 4):
                    out1 = False
                    out2 = True
                else:
                    out1 = False
                    out2 = False
            acc3.append(out1)
            acc4.append(out2)

        acc1 = torch.tensor(acc1).sum() / pred.shape[0]
        acc2 = torch.tensor(acc2).sum() / pred.shape[0]
        acc3 = torch.tensor(acc3).sum() / pred.shape[0]
        acc4 = torch.tensor(acc4).sum() / pred.shape[0]

        return acc1, acc2, acc3, acc4, tmp_out1, tmp_out2

    def run_batch(self, index, pred, true, masks, val):
        eval_masks = index % self.args.pred_len == 0
        if self.args.extra == True:
            pred_ex = pred[masks]
            true_ex = true[masks]
            val_ex = val[masks]
            if true_ex.shape[0] > 0:
                acc1_ex, acc2_ex, acc3_ex, acc4_ex, _, _ = self.check_strategy(pred_ex, true_ex, val_ex, None, None)
                self.total_acc1_ex.append(acc1_ex)
                self.total_acc2_ex.append(acc2_ex)
                self.total_acc3_ex.append(acc3_ex)
                self.total_acc4_ex.append(acc4_ex)

        acc1, acc2, acc3, acc4, tmp_out1, tmp_out2 = self.check_strategy(pred, true, val, eval_masks, index)
        self.total_acc1.append(acc1)
        self.total_acc2.append(acc2)
        self.total_acc3.append(acc3)
        self.strategy_data1 = pd.concat([self.strategy_data1, tmp_out1])
        self.strategy_data2 = pd.concat([self.strategy_data2, tmp_out2])


    def execute_back_test(self, backtest, input_dict):
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


    def back_test_mm(self, trade_data, threshold=10000, num=12):
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

    def back_test_spot_swing(self, trade_data, threshold=15000, num=12):
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
        win = output[output['profit'] > 0].shape[0]

        total = np.round(total / 1000000, 2)
        profit_win = np.round(profit_win / 1000000, 2)
        profit_loss = np.round(profit_loss / 1000000, 2)
        acc_rate = np.round(sum(term) / output.shape[0], 2)
        win_rate = np.round(win / output.shape[0], 2)

        return output, (total, profit_win, profit_loss, acc_rate, win_rate)


    def run(self, epoch):
        acc1 = np.average(self.total_acc1)
        acc2 = np.average(self.total_acc2)
        acc3 = np.average(self.total_acc3)
        acc1_ex = np.average(self.total_acc1_ex)
        acc2_ex = np.average(self.total_acc2_ex)
        acc3_ex = np.average(self.total_acc3_ex)
        acc4_ex = np.average(self.total_acc4_ex)
        strategy_data1 = self.strategy_data1.sort_values(by='date').reset_index(drop=True)
        strategy_data2 = self.strategy_data2.groupby('date').mean().reset_index()
        strategy_data2 = strategy_data2.sort_values(by='date').reset_index(drop=True)

        print(
            "Epoch: {0} ACC1: {1:.5f} ACC2: {2:.5f} ACC3: {3:.5f}  ACC1Ex: {4:.5f} ACC2Ex: {5:.5f} ACC3Ex: {6:.5f} ACC4Ex: {7:.5f}".format(
                epoch + 1, acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, acc4_ex))

        if epoch + 1 >= 10:
            input_dict1 = {'trade_data': strategy_data1, 'num': self.args.pred_len, 'thresh_list': [10000, 15000, 20000]}
            best_output11, values11, dict11 = self.execute_back_test(self.back_test_spot_swing, input_dict1)
            best_output12, values12, dict12 = self.execute_back_test(self.back_test_mm, input_dict1)

            input_dict2 = {'trade_data': strategy_data2, 'num': int(self.args.pred_len/2), 'thresh_list': [10000, 15000, 20000]}
            best_output21, values21, dict21 = self.execute_back_test(self.back_test_spot_swing, input_dict2)
            best_output22, values22, dict22 = self.execute_back_test(self.back_test_mm, input_dict2)

            cnt11 = best_output11.shape[0]
            cnt21 = best_output21.shape[0]
            best_output11.to_csv('best_output11.csv')
            best_output21.to_csv('best_output21.csv')
            strategy_data1.to_csv('strategy_data1.csv')
            strategy_data2.to_csv('strategy_data2.csv')

            print("Test1 | Swing - cnt: {0} best profit: {1} config: {2}  MM bot - best profit: {3} config: {4}".format(
                cnt11, values11, dict11, values12, dict12))
            print("Test2 | Swing - cnt: {0} best profit: {1} config: {2}  MM bot - best profit: {3} config: {4}".format(
                cnt21, values21, dict21, values22, dict22))

        else:
            cnt11 = values11 = dict11 = cnt21 = values21 = dict21 = values12 = dict12 = values22 = dict22 = None

        return acc1, acc2, acc3, acc1_ex, acc2_ex, acc3_ex, acc4_ex, cnt11, values11, dict11, cnt21, values21, dict21, values12, dict12, values22, dict22


def plot_output(file_name, target_col='pred'):
    df = pd.read_csv(file_name)[:10000]
    fig = go.Figure()
    df = df.sort_values(by='date')
    df['date'] = df.date.apply(lambda x:ps.parse(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + ' ' + str(x)[8:10] + ':' + str(x)[10:12]))
    fig.add_trace(go.Scatter(x=df['date'],
                             y=df[target_col],
                             line=dict(color='rgba(17, 1, 1, 1)'),
                             fillcolor='rgba(17, 1, 1, 1)',
                             fill=None,
                             name=target_col))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['hi'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill=None,
                             name='hi'))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['lo'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill='tonexty',
                             name='lo'))


    fig.update_layout(title='推論結果の可視化',
                      plot_bgcolor='white',
                      xaxis=dict(showline=True,
                                 linewidth=1,
                                 linecolor='lightgrey',
                                 tickfont_color='grey',
                                 ticks='inside'),
                      yaxis=dict(title='BTC価格',
                                 showline=True,
                                 linewidth=1,
                                 linecolor='lightgrey',
                                 tickfont_color='grey',
                                 ticks='inside'))
    fig.show()

    return

if __name__ == '__main__':
    from main_yformer import *

    est = Estimation(args)
    file_name = 'strategy_data1.csv'
    data = pd.read_csv(file_name)
    data = data.drop(columns='Unnamed: 0')
    data = data.sort_values(by='date').reset_index(drop=True)
    est.back_test_spot_swing(data)
    plot_output(file_name, target_col='pred')