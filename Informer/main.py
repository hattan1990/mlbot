from exp.exp_informer import Exp_Informer
from config import args
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
            pred.to_excel('output.xlsx')
            spread1.to_excel('spread1.xlsx')
            spread2.to_excel('spread2.xlsx')
            mlflow.log_artifact('output.xlsx')
            mlflow.log_artifact('spread1.xlsx')
            mlflow.log_artifact('spread2.xlsx')

            torch.cuda.empty_cache()

if __name__ == '__main__':
    main(args)