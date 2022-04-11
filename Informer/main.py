from exp.exp_informer import Exp_Informer
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

            pred, spread1, spread2 = exp.predict(setting, True)
            sample_count = spread1.shape[0]
            acc1 = spread1.values[:, 4].sum() / sample_count
            acc2 = spread1.values[:, 8].sum() / sample_count
            acc3 = spread1.values[:, 12].sum() / sample_count
            print("sample_count{0} ACC1:{1:.2f} ACC2:{2:.2f} ACC3:{3:.2f}".format(sample_count, acc1, acc2, acc3))

            pred.to_excel('output.xlsx')
            spread1.to_excel('spread1.xlsx')
            spread2.to_excel('spread2.xlsx')
            mlflow.log_artifact('output.xlsx')
            mlflow.log_artifact('spread1.xlsx')
            mlflow.log_artifact('spread2.xlsx')

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
        else:
            pass

        args_list.append(args)

    return args_list

def update_args_list(args_list, update_name, list):
    add_args = []
    for args in args_list:
        add_args += update_args(args, update_name, list)

    return args_list + add_args

if __name__ == '__main__':
    main(args)

    seq_len_list = [[48, 30, 6],
                    [48, 36, 6],
                    [48, 42, 6]]
    loss_mode_list = ["penalties", "min_max"]
    learning_rate_list = [0.01, 0.05, 0.005]
    drop_out_list = [0.00001, 0.0001, 0.005, 0.001]

    args_list = update_args(args, "seq_len", seq_len_list)
    args_list = update_args_list(args_list, "loss_mode", loss_mode_list)
    args_list = update_args_list(args_list, "learning_rate", learning_rate_list)
    args_list = update_args_list(args_list, "dropout", drop_out_list)

    for i in range(100):
        choice = np.random.choice(len(args_list))
        args_update = args_list[choice]
        main(args_update)