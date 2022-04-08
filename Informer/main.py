from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import mlflow

#ベストパラメーター:pct_change:30）
#{'model': 'informer', 'data': 'GMO-BTCJPY', 'root_path': './dataset/', 'data_path': 'gmo_btcjpy_ohlcv2021.csv', 'features': 'ALL', 'target': 'cl', 'target_num': None, 'freq': 't', 'scaler1': 10000000, 'scaler2': 500, 'checkpoints': './informer_checkpoints', 'seq_len': 48, 'label_len': 24, 'pred_len': 6, 'enc_in': 145, 'dec_in': 1, 'c_out': 1, 'factor': 5, 'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048, 'dropout': 0.0005, 'attn': 'prob', 'embed': 'timeF', 'activation': 'gelu', 'distil': True, 'output_attention': False, 'mix': True, 'padding': 0, 'batch_size': 256, 'learning_rate': 0.001, 'loss': 'mae', 'lradj': 'type1', 'use_amp': False, 'num_workers': 0, 'itr': 1, 'train_epochs': 100, 'patience': 15, 'des': 'exp', 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'detail_freq': 't'}
#Epoch: 10, Steps: 2062 | Train Loss: 0.0012527 Vali Loss: 0.0008630 Spread Loss: 5706.8 Diff_pred_max: 8047.5 Diff_pred_min: 8214.9 Vali_loss local: 8630.4

#その次に期待できる奴（
#{'model': 'informer', 'data': 'GMO-BTCJPY', 'root_path': './dataset/', 'data_path': 'gmo_btcjpy_ohlcv2021.csv', 'features': 'ALL', 'target': 'cl', 'target_num': None, 'freq': 't', 'scaler1': 10000000, 'scaler2': 500, 'checkpoints': './informer_checkpoints', 'seq_len': 96, 'label_len': 48, 'pred_len': 12, 'enc_in': 5, 'dec_in': 1, 'c_out': 1, 'factor': 5, 'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048, 'dropout': 0.0005, 'attn': 'prob', 'embed': 'timeF', 'activation': 'gelu', 'distil': True, 'output_attention': False, 'mix': True, 'padding': 0, 'batch_size': 256, 'learning_rate': 0.01, 'loss': 'mae', 'lradj': 'type1', 'use_amp': False, 'num_workers': 0, 'itr': 1, 'train_epochs': 100, 'patience': 15, 'des': 'exp', 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'detail_freq': 't'}

#モデルを指定します。
title = 'Predict hi&lo'
args = dotdict()
args.runname = 'default'
args.model = 'informer'

#データセットとパスを指定
args.data = 'GMO-BTCJPY'
args.root_path = './dataset/'
args.data_path = 'gmo_btcjpy_ohlcv2021.csv'

#予測タスク、ターゲット(y)、時間フィーチャーエンコーディングを指定
args.features = 'ALL'
args.target = 'cl'
args.target_num = None
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
args.dec_in = 1 # decoder input size
args.c_out = 1 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.0005 # dropout 0.005 ->0.0005

#デフォルトのパラメーター設定
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = False # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0

#バッチサイズ、学習率、ロースファンクションなどを指定
args.batch_size = 256
args.learning_rate = 0.001 #0.0001 -> 0.001
args.loss = 'mae'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

#並列計算するかどうか、トレーニングepoch数を指定
args.num_workers = 0
args.itr = 1
args.train_epochs = 12
args.patience = 15 # 10 -> 15
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
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(title)
    if experiment is not None:
        exp_id = experiment.experiment_id
    else:
        exp_id = client.create_experiment(title)

    tag_list = ['model', 'data', 'root_path', 'data_path', 'features',
                'target', 'target_num', 'freq']
    with mlflow.start_run(experiment_id=exp_id):
        #set tags
        for key in args.keys():
            if key in tag_list:
                mlflow.set_tag(key=key, value=args[key])
            elif key == 'runname':
                mlflow.set_tag(key='mlflow.runName', value=args[key])
            else:
                mlflow.log_param(key, args[key])


        Exp = Exp_Informer

        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                        args.seq_len, args.label_len, args.pred_len,
                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                        args.embed, args.distil, args.mix, args.des, ii)

            exp = Exp(args) # set experiments
            exp.train(setting)

            if args.do_predict:
                exp.predict(setting, True)

            torch.cuda.empty_cache()

if __name__ == '__main__':
    main(args)