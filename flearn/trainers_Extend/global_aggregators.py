import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy

from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, norm_grad_sparse, l2_clip


class BaseGlobalAggregator(object):
    def __init__(self, params):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

    def run(self):
        RuntimeWarning('run function is not implemented')


class SimpleFedAvg(BaseGlobalAggregator):
    def __init__(self, params):
        print('Using simple FedAvg aggregation method.')
        super(SimpleFedAvg, self).__init__(params) 
    
    def run(self, csolns, params):
        if params['gradient_clipping']:
            csolns = l2_clip(csolns)
        
        base = [0] * len(csolns[0])

        for p in csolns:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(csolns) for v in base]

        return averaged_params