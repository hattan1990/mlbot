import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # self.device = self._acquire_device()
        device = self._acquire_device()
        self.model = self._build_model().to(device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.devices == 'mps':
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device(self.args.devices)
                print('Use GPU:{}'.format(device))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def valid(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    