import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from experiments.exp_ETTh import Exp_ETTh

parser = argparse.ArgumentParser(description='SCINet on ETT dataset')


parser.add_argument('--model', type=str, required=False, default='SCINet', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='BTC', choices=['BTC','ETTh1', 'ETTh2', 'ETTm1'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='../dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='GMO_BTC_JPY_ohclv.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='MS', choices=['S', 'M'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='cl', help='target feature')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='exp/ETT_checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--extra', type=bool, default=False)

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')
                                                                                  
### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of SCINet encoder, look back window')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length, horizon')
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)
                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=100, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=320, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =False, help='save the output results')
parser.add_argument('--model_name', type=str, default='SCINet')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)

### -------  model settings --------------  
parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--window_size', default=12, type=int, help='input size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--positionalEcoding', type=bool, default=False)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')
parser.add_argument('--num_decoder_layer', type=int, default=1)
parser.add_argument('--RIN', type=bool, default=False)
parser.add_argument('--decompose', type=bool,default=False)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'BTC': {'data': 'GMO_BTC_JPY_ohclv.csv', 'T': 'cl', 'MS': [5, 1, 1]},
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_ETTh

mae_ = []
maes_ = []
mse_ = []
mses_ = []
def run():
    if args.evaluate:
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, maes, mse, mses = exp.test(setting, evaluate=True)
        print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
    else:
        if args.itr:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr{}'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse,ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

        else:
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)


def run_various_periods():
    for ii in range(args.itr):
        date_range1 = ['2021-04-01 00:00', '2021-05-01 00:00', '2021-06-01 00:00', '2021-07-01 00:00']
        date_range2 = ['2022-04-01 00:00', '2022-05-01 00:00', '2022-06-01 00:00', '2022-07-01 00:00',
                      '2022-08-01 00:00', '2022-09-01 00:00', '2022-10-01 00:00']

        args.data = 'BTC2'

        # Time Range × パラメータ変更（10回）の学習
        for i in range(len(date_range1)):
            for j in range(10):
                seq_lens = [96, 72, 48]
                args.seq_len = np.random.choice(seq_lens)
                label_lens = [48, 36, 24]
                args.label_len = np.random.choice(label_lens)

                n_heads = [8, 12, 16]
                args.n_heads = np.random.choice(n_heads)

                layers = [1, 2, 3]
                args.e_layers = np.random.choice(layers)
                args.d_layers = np.random.choice(layers)

                pred_lens = [12, 30, 40]
                args.pred_len = np.random.choice(pred_lens)

                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr{}'.format(args.model, args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.lr,
                                                                                                      args.batch_size,
                                                                                                      args.hidden_size,
                                                                                                      args.stacks,
                                                                                                      args.levels,
                                                                                                      args.dropout,
                                                                                                      args.inverse, ii)

                args.date_period1 = date_range1[i]
                args.date_period2 = date_range2[i]
                args.date_period3 = date_range2[i+2]

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                print('Time Range from:{} to:{}'.format(args.date_period1, args.date_period3))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                args.date_period2 = date_range2[i + 2]
                args.date_period3 = date_range2[i + 3]
                _ = exp.test(setting, evaluate=True)
                print('Try count :{}'.format(j+1))

                torch.cuda.empty_cache()


if __name__ == '__main__':
    run_various_periods()