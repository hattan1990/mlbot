from exp.exp_informer import Exp_Informer
from utils.tools import dotdict
from config import args
import numpy as np
import torch
import mlflow

title = 'Predict hi&lo eth'

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
            args.best_score = 0
            exp = Exp(args) # set experiments
            exp.test()

            torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main(args)