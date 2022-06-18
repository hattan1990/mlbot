from exp.exp_informer import Exp_Informer
from config import args
import torch
import pandas as pd
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
    pred.to_excel('output.xlsx')
    spread1.to_excel('spread1.xlsx')
    spread2.to_excel('spread2.xlsx')


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

if __name__ == '__main__':
    main(args)
    plot_output()
    plot_spread()