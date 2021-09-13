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


class LocalOpt(BaseLocalOptimizer):
    def __init__(self, params):
        print('Using local SGD only')
        super(LocalOpt, self).__init__(params)
    
    def run(self, c, client_model, local_model, global_model, batches, w_global_idx):
        data_batch = next(batches[c])

        # construct weights
        client_model.set_params(w_global_idx)

        _, grads, _ = c.solve_sgd(data_batch)
        for layer in range(len(grads[1])):
            local_model[layer] = local_model[layer] - self.learning_rate * grads[1][layer]

        loss = c.get_loss() 
        w_global_idx = client_model.get_params()

        return loss, w_global_idx

    def eval(self, fl_env, rd, corrupt_id, batches):
        tmp_models = []
        for idx in range(len(fl_env.clients)):
            tmp_models.append(fl_env.global_model)

        num_train, num_correct_train, loss_vector = fl_env.train_error(tmp_models)
        avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
        num_test, num_correct_test, _ = fl_env.test(tmp_models)
        tqdm.write('At round {} training accu: {}, loss: {}'.format(rd, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
        tqdm.write('At round {} test accu: {}'.format(rd, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
        non_corrupt_id = np.setdiff1d(range(len(fl_env.clients)), corrupt_id)
        tqdm.write('At round {} malicious test accu: {}'.format(rd, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
        tqdm.write('At round {} benign test accu: {}'.format(rd, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
        print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


class DittoOpt(BaseLocalOptimizer):
    def __init__(self, params):
        print('Using local training method introduced in Ditto')
        super(DittoOpt, self).__init__(params)
    
    def run(self, c, client_model, local_model, global_model, batches, w_global_idx):
        # local
        data_batch = next(batches[c])
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

    def eval(self, fl_env, rd, corrupt_id, batches):
        tmp_models = []
        for idx in range(len(fl_env.clients)):
            tmp_models.append(fl_env.local_models[idx])

        num_train, num_correct_train, loss_vector = fl_env.train_error(tmp_models)
        avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
        num_test, num_correct_test, _ = fl_env.test(tmp_models)
        tqdm.write('At round {} training accu: {}, loss: {}'.format(rd, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
        tqdm.write('At round {} test accu: {}'.format(rd, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
        non_corrupt_id = np.setdiff1d(range(len(fl_env.clients)), corrupt_id)
        tqdm.write('At round {} malicious test accu: {}'.format(rd, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
        tqdm.write('At round {} benign test accu: {}'.format(rd, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
        print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


class MetaOpt(BaseLocalOptimizer):
    def __init__(self, params):
        print('Using meta-learning method')
        super(MetaOpt, self).__init__(params)
    
    def run(self, c, client_model, local_model, global_model, batches, w_global_idx):
        c.set_params(w_global_idx)
        data_batch = next(batches[c])
        _, _, _ = c.solve_sgd(data_batch)
        data_batch = next(batches[c])
        grads = c.get_grads(data_batch)

        # approximate Hessian-vector product
        data_batch = next(batches[c])
        delta = 0.001
        dummy_model1 = [(u + delta * v) for u, v in zip(w_global_idx, grads)]
        dummy_model2 = [(u - delta * v) for u, v in zip(w_global_idx, grads)]
        c.set_params(dummy_model1)
        dummy_grad1 = c.get_grads(data_batch)
        c.set_params(dummy_model2)
        dummy_grad2 = c.get_grads(data_batch)
        correction = [(u - v) / (2 * delta) for u, v in zip(dummy_grad1, dummy_grad2)]

        beta = 0.01

        for layer in range(len(grads)):
            w_global_idx[layer] = w_global_idx[layer] - beta * grads[layer] + self.learning_rate * beta * correction[layer]
        
        loss = c.get_loss()
        return loss, w_global_idx
    
    def eval(self, fl_env, rd, corrupt_id, batches):
        tmp_models = []
        for idx in range(len(fl_env.clients)):
            tmp_models.append(fl_env.global_model)

        num_test, num_correct_test, _ = fl_env.test(tmp_models)
        num_train, num_correct_train, loss_vector = fl_env.train_error(tmp_models)

        avg_loss = np.dot(loss_vector, num_train) / np.sum(num_train)

        tqdm.write('At round {} training accu: {}, loss: {}'.format(rd, np.sum(num_correct_train) * 1.0 / np.sum(
            num_train), avg_loss))
        tqdm.write('At round {} test accu: {}'.format(rd, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
        non_corrupt_id = np.setdiff1d(range(len(fl_env.clients)), corrupt_id)
        tqdm.write('At round {} malicious test accu: {}'.format(rd, np.sum(
            num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
        tqdm.write('At round {} benign test accu: {}'.format(rd, np.sum(
            num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
        print("variance of the performance: ",
                np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


class PrivateLayerOpt(BaseLocalOptimizer):
    def __init__(self, params):
        print('Using private layer method')
        super(PrivateLayerOpt, self).__init__(params)
    
    def run(self, c, client_model, local_model, global_model, batches, w_global_idx):
        data_batch = next(batches[c])

        # construct weights
        combine_weight = copy.deepcopy(w_global_idx)
        combine_weight[-1] = local_model[-1]
        client_model.set_params(combine_weight)

        _, grads, _ = c.solve_sgd(data_batch)
        for layer in range(len(grads[1])):
            local_model[layer] = local_model[layer] - self.learning_rate * grads[1][layer]

        loss = c.get_loss() 
        w_global_idx = client_model.get_params()

        return loss, w_global_idx

    def eval(self, fl_env, rd, corrupt_id, batches):
        tmp_models = []
        for idx in range(len(fl_env.clients)):
            combine_weight = copy.deepcopy(fl_env.global_model)
            combine_weight[-1] = fl_env.local_models[idx][-1]
            tmp_models.append(combine_weight)

        num_train, num_correct_train, loss_vector = fl_env.train_error(tmp_models)
        avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
        num_test, num_correct_test, _ = fl_env.test(tmp_models)
        tqdm.write('At round {} training accu: {}, loss: {}'.format(rd, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
        tqdm.write('At round {} test accu: {}'.format(rd, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
        non_corrupt_id = np.setdiff1d(range(len(fl_env.clients)), corrupt_id)
        tqdm.write('At round {} malicious test accu: {}'.format(rd, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
        tqdm.write('At round {} benign test accu: {}'.format(rd, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
        print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))
