import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy


class BaseLocalOptimizer(object):
    def __init__(self, params):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

    def run(self):
        RuntimeWarning('run function is not implemented')


class DittoOpt(BaseLocalOptimizer):
    def __init__(self, params):
        print('Using local training method introduced in Ditto')
        super(DittoOpt, self).__init__(params)
    
    def run(self, c, client_model, local_model, global_model, data_batch, w_global_idx):
        # local
        client_model.set_params(local_model)
        _, grads, _ = c.solve_sgd(data_batch)  


        if self.dynamic_lam:

            model_tmp = copy.deepcopy(local_model)
            model_best = copy.deepcopy(local_model)
            tmp_loss = 10000
            # pick a lambda locally based on validation data
            for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
                for layer in range(len(grads[1])):
                    eff_grad = grads[1][layer] + candidate_lam * (local_model[layer] - global_model[layer])
                    model_tmp[layer] = local_model[layer] - self.learning_rate * eff_grad

                c.set_params(model_tmp)
                l = c.get_val_loss()
                if l < tmp_loss:
                    tmp_loss = l
                    model_best = copy.deepcopy(model_tmp)

            local_model = copy.deepcopy(model_best)

        else:
            for layer in range(len(grads[1])):
                eff_grad = grads[1][layer] + self.lam * (local_model[layer] - global_model[layer])
                local_model[layer] = local_model[layer] - self.learning_rate * eff_grad

        # global
        client_model.set_params(w_global_idx)
        loss = c.get_loss() 
        # losses.append(loss)
        _, grads, _ = c.solve_sgd(data_batch)
        w_global_idx = client_model.get_params()

        return loss, w_global_idx