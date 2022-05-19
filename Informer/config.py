from utils.tools import dotdict
import torch

'''
{'runname': 'feature295', 'model': 'informer', 'loss_mode': 'default', 'data': 'GMO-BTCJPY', 'root_path': './dataset/', 'data_path': 'gmo_btcjpy_ohlcv2021.csv', 'add_feature_num': 60, 'features': 'ALL', 'target': ['hi', 'lo'], 'target_num': None, 'freq': 't', 'scaler1': 10000000, 'scaler2': 500, 'checkpoints': './informer_checkpoints', 'seq_len': 96, 'label_len': 30, 'pred_len': 15, 'enc_in': 295, 'dec_in': 2, 'c_out': 2, 'factor': 5, 'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048, 'dropout': 0.0005, 'attn': 'prob', 'embed': 'timeF', 'activation': 'gelu', 'distil': True, 'output_attention': False, 'mix': True, 'padding': 0, 'batch_size': 256, 'learning_rate': 0.001, 'loss': 'mae', 'lradj': 'type1', 'use_amp': False, 'num_workers': 0, 'itr': 1, 'train_epochs': 20, 'patience': 15, 'des': 'exp', 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'detail_freq': 't'}
ACC : 0.522387
'''

args = dotdict()
args.runname = 'Add Features'
args.model = 'informer'
add_feature_num = 60
args.loss_mode = 'penalties'

#データセットとパスを指定
args.data = 'GMO-BTCJPY'
args.root_path = './dataset/'
args.data_path = 'gmo_btcjpy_ohlcv2021.csv'

#予測タスク、ターゲット(y)、時間フィーチャーエンコーディングを指定
args.add_feature_num = add_feature_num
args.features = 'ALL'
args.target = ['hi', 'lo']
args.target_num = None
args.freq = 't' # h:hourly
args.scaler1 = 10000000 #BTC価格のスケーリング
args.scaler2 = 500 #BTC Volumeのスケーリング

#トレーニング済みモデルを指定
args.checkpoints = './informer_checkpoints'

#EncoderとDecoderの入力するデータの長さを指定
args.seq_len = 96
args.label_len = 36
args.pred_len = 15

#EncoderとDecoderの入力バッチサイズを指定
#モデルのレイア層、self-attentionのヘッド数、全結合層のノード数を指定
args.enc_in = (add_feature_num * 5) + 5 + 250
args.dec_in = 2 # decoder input size
args.c_out = 2 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 16 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.0005 # dropout 0.005 ->0.0005

#デフォルトのパラメーター設定
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
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
args.train_epochs = 20
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
