import numpy as np
from scipy import stats
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


class KrumAgg(BaseGlobalAggregator):
    def __init__(self, params):
        print('Using Krum aggregation method.')
        super(KrumAgg, self).__init__(params)
    
    def run(self, csolns, params):
        if params['gradient_clipping']:
            csolns = l2_clip(csolns)
        
        k = params['expected_num_benign'] - 2

        flattened_grads = []
        for i in range(len(csolns)):
            flattened_grads.append(process_grad(csolns[i]))
        distance = np.zeros((len(csolns), len(csolns)))
        for i in range(len(csolns)):
            for j in range(i+1, len(csolns)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(csolns))
        for i in range(len(csolns)):
            score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

        selected_idx = np.argsort(score)[0]

        return csolns[selected_idx]


class MultiKrumAgg(BaseGlobalAggregator):
    def __init__(self, params):
        print('Using Multi Krum aggregation method.')
        self.fedavg = SimpleFedAvg(params)
        super(MultiKrumAgg, self).__init__(params)
    
    def run(self, csolns, params):
        if params['gradient_clipping']:
            csolns = l2_clip(csolns)
            params['gradient_clipping'] = False
        
        m = params['expected_num_benign']
        k = params['expected_num_benign'] - 2

        flattened_grads = []
        for i in range(len(csolns)):
            flattened_grads.append(process_grad(csolns[i]))
        distance = np.zeros((len(csolns), len(csolns)))
        for i in range(len(csolns)):
            for j in range(i + 1, len(csolns)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(csolns))
        for i in range(len(csolns)):
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  # the distance including itself, so k+1 not k

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(csolns[i])

        return self.fedavg.run(selected_parameters, params)


class ElementWiseMedian(BaseGlobalAggregator):
    def __init__(self, params):
        print('Using element-wise median based aggregation method.')
        super(ElementWiseMedian, self).__init__(params)
    
    def run(self, csolns, params):
        if params['gradient_clipping']:
            csolns = l2_clip(csolns)

        num_layers = len(csolns[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(csolns)):
                a.append(csolns[j][i].flatten())
            aggregated_models.append(np.reshape(np.median(a, axis=0), newshape=csolns[0][i].shape))

        return aggregated_models


class ElementWiseTrimmedMean(BaseGlobalAggregator):
    def __init__(self, params):
        print('Using element-wise trimmed mean based aggregation method.')
        super(ElementWiseTrimmedMean, self).__init__(params)
    
    def run(self, csolns, params):
        if params['gradient_clipping']:
            csolns = l2_clip(csolns)

        num_layers = len(csolns[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(csolns)):
                a.append(csolns[j][i].flatten())
            
            a = np.array(a)
            trimmed_mean = stats.trim_mean(a, self.trim_ratio)

            aggregated_models.append(np.reshape(trimmed_mean, newshape=csolns[0][i].shape))

        return aggregated_models