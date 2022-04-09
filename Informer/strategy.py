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
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                    args.embed, args.distil, args.mix, args.des, 0)

    exp = Exp(args) # set experiments
    pred, spread1, spread2 = exp.predict(setting)
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
                             y=df['t_min'],
                             line=dict(color='rgba(17, 250, 244, 0.5)'),
                             fillcolor='rgba(17, 250, 244, 0.5)',
                             fill='tonexty',
                             name='t_min'))

    fig.add_trace(go.Scatter(x=df['date'],
                             y=df['p_max'],
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
    #plot_output()
    #plot_spread()