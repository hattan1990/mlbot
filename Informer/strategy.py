from exp.exp_informer import Exp_Informer
from config import args
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main(args):
    print('Args in experiment:')
    print(args)

    Exp = Exp_Informer
    exp = Exp(args) # set experiments
    pred, spread1, spread2 = exp.predict()
    sample_count = spread1.shape[0]
    acc1 = spread1.values[:, 4].sum() / sample_count
    acc2 = spread1.values[:, 8].sum() / sample_count
    acc3 = spread1.values[:, 12].sum() / sample_count
    print("sample_count{0} ACC1:{1:.2f} ACC2:{2:.2f} ACC3:{3:.2f}".format(sample_count, acc1, acc2, acc3))
    pred.to_excel('output_v1.xlsx')
    spread1.to_excel('spread1_v1.xlsx')
    spread2.to_excel('spread2_v1.xlsx')


    return pred

def plot_output():
    df = pd.read_excel('output.xlsx')
    target_col = 'pred'
    fig = go.Figure()
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

def plot_spread():
    df = pd.read_excel('spread2.xlsx')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['t_max'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill=None,
                             name='t_max'))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['t_min']+1000,
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill='tonexty',
                             name='t_min'))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['p_max']-1000,
                             line=dict(color='rgba(17, 250, 1, 1)'),
                             fillcolor='rgba(17, 250, 1, 1)',
                             fill=None,
                             name='p_max'))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['p_min'],
                             line=dict(color='rgba(17, 250, 1, 1)'),
                             fillcolor='rgba(17, 250, 1, 1)',
                             fill='tonexty',
                             name='p_min'))

    check = pd.read_excel('spread1.xlsx')
    check_df = check[check[8] == 1][[0, 5]]
    check_df = check_df.rename(columns={0:'date', 5:'text'})
    check_df = pd.merge(check_df, df, on='date', how='left')
    fig.add_trace(go.Scatter(x=check_df['date'],
                             y=(check_df['p_max']+check_df['p_min'])/2,
                             mode='markers',
                             line=dict(color='#FF0000'),
                             marker=dict(color='#FF0000', size=10, opacity=0.8, symbol='star'),
                             text=check_df['text'],
                             name='profit'))


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

def back_test_megin_swing(version='v1'):
    trade_data = pd.read_excel('output_' + version + '.xlsx')

    output = []
    stocks = []
    total = 0
    trade_cnt = 0
    for i in range(0, trade_data.shape[0], 12):
        if i == 0:
            start = 0
            end = 12 - 1
        else:
            end = i + 12 - 1

        tmp_data = trade_data.loc[start:end]
        pred_spread_min = int(tmp_data['pred'].min())
        pred_spread_max = int(tmp_data['pred'].max())
        spread_mergin = (pred_spread_max - pred_spread_min)

        buy = False
        sell = False

        if spread_mergin >= 15000:
            trade_cnt += 1
            for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
                if hi > pred_spread_max:
                    sell = True
                    sell_price = pred_spread_max
                    sell_date = date
                if lo < pred_spread_min:
                    buy = True
                    buy_price = pred_spread_min
                    buy_date = date


        close_price = tmp_data['cl'].values[-1]
        close_date = tmp_data['date'].values[-1]

        if (sell == True) & (buy == True):
            profit = sell_price - buy_price
        elif (sell == True) & (buy == False):
            profit = sell_price - close_price
            stocks.append([sell_date, sell_price])
        elif (sell == False) & (buy == True):
            profit = close_price - buy_price
            stocks.append([buy_date, buy_price])
        else:
            profit = 0
            pass
        total += profit
        if profit != 0:
            print('TOTAL:{} date:{} profit:{} buy:{} sell:{} stoks:{} tradeCount:{}'.format(total, close_date, profit, buy, sell, len(stocks), trade_cnt))
        output.append([close_date, total, profit, buy, sell])
        start = end + 1

    return pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell'])

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

def back_test_mm(version='v1'):
    trade_data = pd.read_excel('output_' + version + '.xlsx')

    output = []
    buy_stocks = []
    sell_stocks = []
    stock_counts = []
    total = 0
    trade_cnt = 0
    max_stocks = 0
    for i in range(0, trade_data.shape[0], 12):
        if i == 0:
            start = 0
            end = 12 - 1
        else:
            end = i + 12 - 1

        tmp_data = trade_data.loc[start:end]
        pred_spread_min = int(tmp_data['pred'].min())
        pred_spread_max = int(tmp_data['pred'].max())
        spread_mergin = (pred_spread_max - pred_spread_min)

        buy = False
        sell = False
        trade = False

        for date, hi, lo in tmp_data[['date', 'hi', 'lo']].values:
            buy_stocks = drop_off_buy_stocks(buy_stocks,  hi)
            sell_stocks = drop_off_sell_stocks(sell_stocks, lo)

            if spread_mergin >=10000:
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
            print('TOTAL:{} MaxStocks:{} date:{} profit:{} buy:{} sell:{} stoks:{} tradeCount:{}'.format(total, max_stocks, close_date, profit,
                                                                                            buy, sell, stocks_count,
                                                                                            trade_cnt))
        else:
            pass

        output.append([close_date, total, profit, buy, sell])
        start = end + 1

    stock_df = pd.DataFrame(buy_stocks+sell_stocks, columns=['date', 'price', 'stay_cnt', 'drop_off', 'diff'])
    print("stock count mean {}".format(np.mean(stock_counts)))

    return pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell']), stock_df

def back_test_spot_swing(threshold=15000, version='v1', pred_opsion=''):
    trade_data = pd.read_excel('output_' + version + '.xlsx')

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
        elif pred_opsion == 'mean2':
            spread = tmp_data['pred_hi'].max() - tmp_data['pred_lo'].min()
            pred_spread_min = int(tmp_data['pred_lo'].min() + spread/4)
            pred_spread_max = int(tmp_data['pred_hi'].max() - spread/4)

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
        if profit != 0:
            print('TOTAL:{} date:{} profit:{} buy:{} sell:{} tradeCount:{}'.format(total, close_date, profit, buy, sell, trade_cnt))
        output.append([close_date, total, profit, buy, sell])
        start = end + 1

    return pd.DataFrame(output, columns=['date', 'total', 'profit', 'buy', 'sell'])

if __name__ == '__main__':
    #main(args)
    #output = back_test_megin_swing(version='v1')
    #plot_output()
    #plot_spread()
    #output, stocks = back_test_mm(version='v1')
    #print(stocks)
    #output.to_excel('back_test_megin_swing.xlsx')

    output = back_test_spot_swing(threshold=10000, version='v1_0711', pred_opsion='zero')
    #output.to_excel('ck_old.xlsx')