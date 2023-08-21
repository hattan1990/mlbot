import os
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
                columns = ['date', 'op', 'hi', 'lo', 'cl', 'true', 'pred']

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
            output, scores = backtest(trade_data, rate=thresh, num=num)
            if (scores[0] > best_score) or (i == 0):
                best_score = scores[0]
                best_output = output
                best_score_values = scores
                out_dict = {'vesion': num, 'thresh': thresh}

        return best_output, best_score_values, out_dict


    def back_test_mm(self, trade_data, rate=3, num=12):
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
        stocks_count = 0
        max_stocks = 0
        for i in range(0, trade_data.shape[0], num):
            if i == 0:
                start = 0
                end = num - 1
            else:
                end = i + num - 1

            tmp_data = trade_data.loc[start:end]

            preds = tmp_data['pred'].values
            pred_spread_min = int(np.percentile(preds, 20))
            pred_spread_max = int(np.percentile(preds, 80))

            threshold = rate

            buy = False
            sell = False
            trade = False

            for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                buy_stocks = drop_off_buy_stocks(buy_stocks, hi)
                sell_stocks = drop_off_sell_stocks(sell_stocks, lo)

                if (stocks_count < threshold):
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

            output.append([close_date, total, profit, buy, sell, stocks_count])
            start = end + 1

        output = pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell', 'stocks_count'])
        output = output[output['profit'] != 0]
        term = (output['buy'] == True) & (output['sell'] == True)
        profit_win = output.loc[term, 'profit'].sum()

        stock_mean = np.mean(stock_counts)

        if stock_mean < 1:
            stock_mean = 1
        total = np.round(total / stock_mean, 2) / 1000000
        profit_win = np.round(profit_win / stock_mean, 2) / 1000000

        return output, (total, profit_win, stock_mean, max_stocks)


    def fix_stocks(self, position, stock_count, price_list, price2):
        if position == 'short':
            for price1 in price_list:
                if price1 >= price2:
                    price_list.remove(price1)
                    stock_count -= 1
                else:
                    pass
        else:
            for price1 in price_list:
                if price1 <= price2:
                    price_list.remove(price1)
                    stock_count -= 1
                else:
                    pass

        return stock_count

    def back_test_spot_swing(self, trade_data, rate=0.002, num=12, max_stock=3):
        output = []
        total = 0
        trade_cnt = 0
        stock_count = 0
        long_prices = []
        short_prices = []
        for i in range(0, trade_data.shape[0], num):
            if i == 0:
                start = 0
                end = num - 1
            else:
                end = i + num - 1

            tmp_data = trade_data.loc[start:end]
            base_price = tmp_data['op'].values[0]
            #pred_spread_min = int(tmp_data['pred'].min())
            #pred_spread_max = int(tmp_data['pred'].max())
            pred_spread_mean = int(tmp_data['pred'].mean())
            spread_to_max = (pred_spread_mean - base_price)
            spread_to_min = (base_price - pred_spread_mean)

            threshold = int(tmp_data['pred'].mean()) * rate
            buy = False
            sell = False

            if (spread_to_max <= threshold) & (spread_to_max > spread_to_min)&(stock_count <= max_stock):
                trade_cnt += 1
                buy = True
                buy_price = base_price
                for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                    if hi > pred_spread_mean:
                        sell = True
                        sell_price = pred_spread_mean

            elif (spread_to_min <= threshold) & (spread_to_max < spread_to_min)&(stock_count <= max_stock):
                trade_cnt += 1
                sell = True
                sell_price = base_price
                for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                    if lo < pred_spread_mean:
                        buy = True
                        buy_price = pred_spread_mean

            close_price = tmp_data['cl'].values[-1]
            close_date = tmp_data['date'].values[-1]
            hi_price = tmp_data['hi'].values[-1]
            lo_price = tmp_data['lo'].values[-1]

            if len(long_prices) > 0:
                stock_count = self.fix_stocks('long', stock_count, long_prices, hi_price)
            elif len(short_prices) > 0:
                stock_count = self.fix_stocks('short', stock_count, short_prices, lo_price)
            else:
                pass

            if (sell == True) & (buy == True):
                profit = sell_price - buy_price
            elif (sell == True) & (buy == False):
                #profit = sell_price - close_price
                profit = 0
                stock_count += 1
                short_prices.append(sell_price)
            elif (sell == False) & (buy == True):
                #profit = close_price - buy_price
                profit = 0
                stock_count += 1
                long_prices.append(buy_price)
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
        acc_rate = np.round(sum(term) / (output.shape[0]+1), 2)
        win_rate = np.round(win / (output.shape[0]+1), 2)

        return output, (total, profit_win, profit_loss, acc_rate, win_rate)


    def back_test_spot_doten(self, trade_data, rate=0.002, num=12):
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
            preds = tmp_data['pred'].values
            pred_spread_min = int(np.percentile(preds, 20))
            pred_spread_max = int(np.percentile(preds, 80))
            spread_to_max = (pred_spread_max - base_price)
            spread_to_min = (base_price - pred_spread_min)

            buy = False
            sell = False
            info = ''
            close_price = tmp_data['cl'].values[-1]
            close_date = tmp_data['date'].values[-1]

            if (spread_to_max > spread_to_min):
                trade_cnt += 1
                buy = True
                buy_price = base_price
                for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                    if hi > pred_spread_max:
                        sell = True
                        sell_price = pred_spread_max
                        info = 'Success-LONG'
                    elif lo < pred_spread_min:
                        close_price = pred_spread_min
                        info = 'Miss-LONG-LossCut'
                    else:
                        info = 'Miss-LONG'
                        pass

            elif (spread_to_max < spread_to_min):
                trade_cnt += 1
                sell = True
                sell_price = base_price
                for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                    if lo < pred_spread_min:
                        buy = True
                        buy_price = pred_spread_min
                        info = 'Success-SHORT'
                    elif hi > pred_spread_max:
                        close_price = pred_spread_max
                        info = 'Miss-SHORT-LossCut'
                    else:
                        info = 'Miss-SHORT'
                        pass

            if (sell == True) & (buy == True):
                profit = sell_price - buy_price
                str_val = str(sell_price) + '|' + str(buy_price)
            elif (sell == True) & (buy == False):
                profit = sell_price - close_price
                str_val = str(sell_price) + '|' + str(close_price)
            elif (sell == False) & (buy == True):
                profit = close_price - buy_price
                str_val = str(close_price) + '|' + str(buy_price)
            else:
                profit = 0
                str_val = ''
                pass
            total += profit
            output.append([close_date, info, total, profit, str_val, buy, sell])
            start = end + 1

        output = pd.DataFrame(output, columns=['date', 'info', 'total', 'profit', 'sell&buy_price', 'buy', 'sell'])
        output = output[output['profit'] != 0]
        term = (output['buy'] == True) & (output['sell'] == True)
        profit_win = output.loc[term, 'profit'].sum()
        profit_loss = output.loc[~term, 'profit'].sum()
        win = output[output['profit'] > 0].shape[0]

        total = np.round(total / 1000000, 2)
        profit_win = np.round(profit_win / 1000000, 2)
        profit_loss = np.round(profit_loss / 1000000, 2)
        acc_rate = np.round(sum(term) / (output.shape[0]+1), 2)
        win_rate = np.round(win / (output.shape[0]+1), 2)

        return output, (total, profit_win, profit_loss, acc_rate, win_rate)


    def run(self, epoch, target_time_range_from, target_time_range_to):
        self.trade_range_from = target_time_range_from
        self.trade_range_to = target_time_range_to
        acc1 = np.average(self.total_acc1)
        acc2 = np.average(self.total_acc2)
        acc3 = np.average(self.total_acc3)
        strategy_data1 = self.strategy_data1.sort_values(by='date').reset_index(drop=True)

        if epoch + 1 >= 5:
            print(
                "Epoch: {0} ACC1: {1:.5f} ACC2: {2:.5f} ACC3: {3:.5f} ".format(
                    epoch + 1, acc1, acc2, acc3))
            input_dict1 = {'trade_data': strategy_data1, 'num': self.args.pred_len, 'thresh_list': [3, 5, 7]}

            best_output11, values11, dict11 = self.execute_back_test(self.back_test_mm, input_dict1)
            best_output12, values12 = self.back_test_spot_swing(strategy_data1, rate=0.002, num=self.args.pred_len, max_stock=3)
            best_output13, values13 = self.back_test_spot_doten(strategy_data1, rate=0.002, num=self.args.pred_len)

            best_output11['month'] = best_output11['date'].apply(lambda x: str(x)[:6])

            cnt11 = best_output11.shape[0]
            cnt12 = best_output12.shape[0]
            cnt13 = best_output13.shape[0]

            if self.args.data_path == 'GMO_BTC_JPY_ohclv_eval.csv':
                os.mkdir(str(acc1))
                best_output11.to_csv(str(acc1) + '/best_output11.csv')
                best_output12.to_csv(str(acc1) + '/best_output12.csv')
                best_output13.to_csv(str(acc1) + '/best_output13.csv')
                strategy_data1.to_csv(str(acc1) + '/strategy_data1.csv')
            else:
                best_output11.to_csv('best_output11.csv')
                best_output12.to_csv('best_output12.csv')
                best_output13.to_csv('best_output13.csv')
                strategy_data1.to_csv('strategy_data1.csv')

            print("Test1 | MM - cnt: {0} best profit: {1} config: {2} ".format(
                cnt11, values11, dict11))

            print("Test2 | Spot Swing - cnt: {0} best profit: {1}".format(
                cnt12, values12))

            print("Test3 | Spot Doten - cnt: {0} best profit: {1}".format(
                cnt13, values13))


        else:
            cnt11 = values11 = dict11 = None

        return acc1, acc2, acc3, cnt11, values11, dict11, strategy_data1

def calc_mergin_pred(df, args):
    output = pd.DataFrame()
    for i in range(0, df.shape[0], args.pred_len - 1):
        if i == 0:
            from_num = 0
            to_num = args.pred_len - 1
            tmp_df = df.loc[from_num:to_num, :]
            from_num = to_num + 1
        else:
            to_num = from_num + args.pred_len - 1
            tmp_df = df.loc[from_num:to_num, :]
            from_num = to_num + 1

        if tmp_df.shape[0] > 0:
            lo = tmp_df['lo'].min()
            hi = tmp_df['hi'].max()
            date = tmp_df['date'].max()
            pred_min = tmp_df['pred'].min()
            pred_max = tmp_df['pred'].max()
            tmp_dic = {'date':date, 'lo':lo, 'hi':hi, 'pred_min':pred_min, 'pred_max':pred_max}
            tmp_out = pd.DataFrame([tmp_dic])
            output = pd.concat([output, tmp_out])

    return output.reset_index(drop=True)

def plot_mergin(file_name, args):
    df = pd.read_csv(file_name)
    df['date'] = df.date.apply(lambda x: ps.parse(
        str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + ' ' + str(x)[8:10] + ':' + str(x)[10:12]))
    df = df.sort_values(by='date')
    plot_df = calc_mergin_pred(df, args)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['hi'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill=None,
                             name='hi'))

    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['lo'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill='tonexty',
                             name='lo'))

    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['pred_max'],
                             line=dict(color='rgba(17, 1, 1, 0.5)'),
                             fillcolor='rgba(17, 1, 1, 0.5)',
                             fill=None,
                             name='pred_hi'))

    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['pred_min'],
                             line=dict(color='rgba(17, 1, 1, 0.5)'),
                             fillcolor='rgba(17, 1, 1, 0.5)',
                             fill='tonexty',
                             name='pred_lo'))

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

def plot_output(file_name, args):
    target_col = 'pred'
    if file_name == 'strategy_data1.csv':
        target = args.pred_len
    else:
        target = (args.pred_len / 2)
    df = pd.read_csv(file_name)
    df['date'] = df.date.apply(lambda x:ps.parse(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + ' ' + str(x)[8:10] + ':' + str(x)[10:12]))
    df = df.sort_values(by='date')
    fig = go.Figure()
    target_index_list = []
    for i, values in enumerate(df.values):
        target_index = i % target
        target_index_list.append(target_index)
    df['target_index'] = target_index_list

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

    index_df = df[df['target_index'] == (target-1)]
    fig.add_trace(go.Scatter(x=index_df['date'],
                             y=index_df['pred'],
                             mode='markers',
                             name='index'))

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
    from run_SCINet import *

    args.pred_len = 15
    est = Estimation(args)
    file_name = 'strategy_data1.csv'
    data = pd.read_csv(file_name)
    data = data.drop(columns='Unnamed: 0')
    data['date'] = data.date.apply(lambda x: ps.parse(
        str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + ' ' + str(x)[8:10] + ':' + str(x)[10:12]))
    data = data.sort_values(by='date').reset_index(drop=True)
    output = est.back_test_spot_swing(data, rate=0.002, num=args.pred_len)
    output = est.back_test_mm(data, rate=2, num=args.pred_len)
    output2 = est.back_test_spot_swing(data, rate=0.0005, num=15)
    print(output[0].shape[0], output[1])
    print(output2[0].shape[0], output2[1])
    output[0].to_excel('output.xlsx')
    output2[0].to_excel('output2.xlsx')
    #plot_mergin(file_name, args)