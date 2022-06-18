from exp.exp_informer import Exp_Informer
from utils.tools import dotdict
from config import args
import numpy as np
import torch
import mlflow

title = 'Predict hi&lo'

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

            torch.cuda.empty_cache()

def update_args(input_args, update_name, list):
    args_list = []
    for values in list:
        args = input_args.copy()
        if update_name == 'seq_len':
            args['seq_len'] = values[0]
            args['label_len'] = values[1]
            args['pred_len'] = values[2]
        elif update_name == 'loss_mode':
            args['loss_mode'] = values
        elif update_name == 'learning_rate':
            args['learning_rate'] = values
        elif update_name == 'dropout':
            args['dropout'] = values
        elif update_name == 'n_heads':
            args['n_heads'] = values
        else:
            pass

        args_list.append(args)

    return args_list

def update_args_list(args_list, update_name, list):
    add_args = []
    for args in args_list:
        add_args += update_args(args, update_name, list)

    return args_list + add_args

def validation(args_list):
    for i in range(10):
        choice = np.random.choice(len(args_list))
        args_update = args_list[choice]
        args_update = dotdict(args_update)
        main(args_update)

if __name__ == '__main__':
    seq_len_list = [[72, 36, 12],[72, 24, 12], [96, 36, 12]]
    loss_mode_list = ["default"]
    n_heads_list = [4, 8, 12, 16]
    args_list = update_args(args, "seq_len", seq_len_list)
    args_list = update_args_list(args_list, "loss_mode", loss_mode_list)
    args_list = update_args_list(args_list, "n_heads", n_heads_list)
    validation(args_list)