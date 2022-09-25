
import time

import numpy as np
import torch


from torch.optim import Adam
from torch.utils.data import DataLoader


import torch.nn as nn

import ipc
from config import build_parser
from model import Transformer
from data_loader import Dataset_BTC
from metrics import metric

from strategy import Estimation


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, enc_attn_type=args.encoder_attention,
                       dec_attn_type=args.decoder_attention, dropout=args.dropout)


def get_params(mdl):
    return mdl.parameters()


def _get_data(args, flag):
    Data = Dataset_BTC

    if flag == 'val':
        shuffle_flag = False;
        drop_last = True;
        batch_size = 32
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
        # freq = args.detail_freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        # freq = args.freq

    data_set = Data(
        root_path='../dataset/',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def run_metrics(caption, preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}'.format(caption, mse, mae))
    return mse, mae


def run_iteration(model, loader, args, training=True, message = ''):
    preds = []
    trues = []
    inds = []
    evals = []
    total_loss = 0
    elem_num = 0
    steps = 0
    if args.device != 'cpu':
        target_device = 'cuda:{}'.format(args.local_rank)
    else:
        target_device = args.device
    for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_eval) in enumerate(loader):
        if not args.deepspeed:
            model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32,
                              device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)

        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')

        pred = result.detach().cpu().numpy()  # .squeeze()
        true = target.detach().cpu().numpy()  # .squeeze()
        ind = index
        eval = batch_eval


        preds.append(pred)
        trues.append(true)
        inds.append(ind)
        evals.append(eval)

        unscaled_loss = loss.item()
        total_loss += unscaled_loss
        if training:
            if args.deepspeed:
                from deepspeed import deepspeed
                model.backward(loss)
                model.step()
            else:
                loss.backward()
                model.optim.step()

            return preds, trues

        else:
            return inds, preds, trues, evals

def inverse_transform_batch(batch_values, scaler):
    output = []
    for values in batch_values:
        out = scaler.inverse_transform(values)
        output.append(out)
    return np.array(output)

def preform_experiment(args):
    model = get_model(args)
    params = list(get_params(model))
    print('Number of parameters: {}'.format(len(params)))

    if args.deepspeed:
        from deepspeed import deepspeed
        deepspeed_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                              model=model,
                                                              model_parameters=params)
    else:
        model.to(args.device)
        model.optim = Adam(params, lr=0.001)

    train_data, train_loader = _get_data(args, flag='train')
    assert len(train_data.data_x[0]) == args.input_len, \
        "Dataset contains input vectors of length {} while input_len is set to {}".format(len(train_data.data_x[0], args.input_len))
    assert len(train_data.data_y[0]) == args.output_len, \
        "Dataset contains output vectors of length {} while output_len is set to {}".format(
            len(train_data.data_y[0]), args.output_len)

    start = time.time()
    for iter in range(1, args.iterations + 1):
        preds, trues = run_iteration(deepspeed_engine if args.deepspeed else model , train_loader, args, training=True, message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        mse, mae = run_metrics("Loss after iteration {}".format(iter), preds, trues)

    print('cuda max memory allocated : {}'.format(torch.cuda.max_memory_allocated()))

    if args.debug:
        model.record()

    test_data, test_loader = _get_data(args, flag='val')
    if args.deepspeed:
        model.inference()
    else:
        model.eval()
    # Model evaluation on validation data
    estimation = Estimation(args)
    inds, v_preds, v_trues, evals = run_iteration(deepspeed_engine if args.deepspeed else model, test_loader, args, training=False, message="Validation set")
    mse, mae = run_metrics("Loss for validation set ", v_preds, v_trues)

    scaler = train_data.scaler_target
    total_loss_real = []
    for index, pred_batch, true_batch, batch_eval in zip(inds, v_preds, v_trues, evals):
        pred_real = inverse_transform_batch(pred_batch, scaler)
        true_real = inverse_transform_batch(true_batch, scaler)
        loss_real = np.mean(abs(pred_real - true_real))
        total_loss_real.append(loss_real)

        # Strategyモジュール追加
        batch_eval = batch_eval[:, -args.pred_len:, :].to(device)
        masks = _create_masks(pred_real, batch_eval)
        estimation.run_batch(index, pred_real, true_real, masks, batch_eval)

    total_loss_real = np.average(total_loss_real)
    print('MSE: {}, MAE: {}, Real Loss: {}'.format(mse, mae, total_loss_real))
    estimation.run(99)

def _create_masks(batch_y, batch_val, mergin=20000):
    masks = []
    for hi_lo, val in zip(batch_y, batch_val):
        op = val[0, 1]
        hi_max = hi_lo[: ,0].max()
        lo_min = hi_lo[: ,0].min()
        spread1 = (hi_max - op)
        spread2 = (op - lo_min)
        if (spread1 >= mergin) or (spread2 >= mergin):
            masks.append(True)
        else:
            masks.append(False)

    return torch.tensor(masks)

def main(deepspeed_flg, device):
    parser = build_parser(deepspeed_flg)
    args = parser.parse_args(None)
    args.device=device
    preform_experiment(args)


if __name__ == '__main__':
    deepspeed_flg = False
    device = 'cpu' #[cuda:0, cpu]
    main(deepspeed_flg, device)


