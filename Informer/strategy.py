from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#モデルを指定します。
args = dotdict()
args.model = 'informer'

#データセットとパスを指定
args.data = 'GMO-BTCJPY'
args.root_path = './dataset/'
args.data_path = 'gmo_btcjpy_ohlcv_val.csv'

#予測タスク、ターゲット(y)、時間フィーチャーエンコーディングを指定
args.features = 'ALL'
args.target = ['hi', 'lo']
args.freq = 't' # h:hourly
args.scaler1 = 10000000 #BTC価格のスケーリング
args.scaler2 = 500 #BTC Volumeのスケーリング

#トレーニング済みモデルを指定
args.checkpoints = './informer_checkpoints'

#EncoderとDecoderの入力するデータの長さを指定
args.seq_len = 96
args.label_len = 48
args.pred_len = 12

#EncoderとDecoderの入力バッチサイズを指定
#モデルのレイア層、self-attentionのヘッド数、全結合層のノード数を指定
args.enc_in = 145 # encoder input size
args.dec_in = 2 # decoder input size
args.c_out = 2 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout

#デフォルトのパラメーター設定
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = False # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0

#バッチサイズ、学習率、ロースファンクションなどを指定
args.batch_size = 1
args.learning_rate = 0.0001
args.loss = 'mae'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

#並列計算するかどうか、トレーニングepoch数を指定
args.num_workers = 0
args.itr = 1
args.train_epochs = 6
args.patience = 10
args.des = 'exp'

#GPUを使用するかどうかを指定
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

#トレーニングデータセットのパラメータを設定
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

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