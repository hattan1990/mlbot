import argparse
import os
import torch
import numpy as np
from experiments.exp_LaST import Exp_LaST

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='LaST for TSF')

# -------  dataset settings --------------
parser.add_argument('--data', type=str, default='BTC',
                    choices=['BTC', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', "Exchange_rate", "Electricity", "Weather"],
                    help='name of dataset')
parser.add_argument('--root_path', type=str, default='../dataset/',
                    choices=['./datasets/ETT-data/', './datasets/'], help='root path of the data file')
parser.add_argument('--data_path', type=str, default=' GMO_BTC_JPY_1min_ohclv.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='MS',choices=['S', 'M', 'MS'],
                    help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='exp/checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default=False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# -------  input/output length settings --------------
parser.add_argument('--seq_len', type=int, default=201, help='input sequence length of encoder, look back window')
parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=120, help='prediction sequence length, horizon')

# -------  model settings --------------
parser.add_argument('--model', type=str, required=False, default='LaST', help='model of the experiment')
parser.add_argument('--latent_size', default=128, type=int, help='latent size of model')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

# -------  training settings --------------
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=320, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae', help='loss function')
parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
parser.add_argument('--model_name', type=str, default='LaST')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--seed', type=int, default=4321, help='random seed')


def main():
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() | torch.backends.mps.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'BTC'  : {'data': ' GMO_BTC_JPY_1min_ohclv.csv', 'T': 'cl', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 1, 1]},
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'Exchange_rate': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
        'Electricity': {'data': 'electricity.csv', 'T': 'MT_369', 'M': [321, 321, 321], 'S': [1, 1, 1],
                        'MS': [321, 321, 1]},
        'Weather': {'data': 'weather.csv', 'T': 'OT', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp = Exp_LaST


    for ii in range(args.itr):
        date_range1 = ['2022-01-01 00:00', '2022-02-01 00:00']
        date_range2 = ['2022-12-01 00:00', '2023-01-01 00:00', '2023-02-01 00:00', '2023-03-01 00:00',
                       '2023-04-01 00:00', '2023-05-01 00:00']

        # Time Range × パラメータ変更（10回）の学習
        for i in range(len(date_range1)):
            for j in range(3):
                args.option = 0

                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_hid{}_s{}_op{}_nh{}'.format(args.model, args.data,
                                                                                 args.features,
                                                                                 args.seq_len,
                                                                                 args.label_len,
                                                                                 args.pred_len,
                                                                                 args.hidden_size,
                                                                                 args.stacks,
                                                                                 args.option,
                                                                                 args.n_heads)

                args.date_period1 = date_range1[0]
                args.date_period2 = date_range2[i]
                args.date_period3 = date_range2[i + 3]

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                print('Time Range from:{} to:{}'.format(args.date_period1, args.date_period3))
                exp.train(setting)

                if args.date_period3 != '2022-11-01 00:00':
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    args.date_period2 = date_range2[i + 3]
                    args.date_period3 = date_range2[i + 4]
                    _ = exp.test(setting, evaluate=True)
                    print('Try count :{}'.format(j + 1))

                torch.cuda.empty_cache()
if __name__ == '__main__':
    main()