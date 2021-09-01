import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, norm_grad_sparse, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba
from flearn.utils.language_utils import letter_to_vec, word_to_indices


class Server(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)
        
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.dynamic_lam, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = copy.deepcopy(self.client_model.get_params())  # self.latest_model is the global model
        self.local_models = []
        self.interpolation = []
        self.global_model = copy.deepcopy(self.latest_model)
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))
            self.interpolation.append(copy.deepcopy(self.latest_model))

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, dynamic_lam, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], dynamic_lam, model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self, models):
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.train_error()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test(self, models):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def validate(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(self.local_models[idx])
            ct, ns = c.validate()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(tot_correct)


    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients

        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round + 4)

        if self.sampling == 1:
            pk = np.ones(num_clients) / num_clients
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

        elif self.sampling == 2:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]


    def aggregate(self, wsolns):

        total_weight = 0.0
        base = [0] * len(wsolns[0][1])

        for (w, soln) in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def simple_average(self, parameters):

        base = [0] * len(parameters[0])

        for p in parameters:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(parameters) for v in base]

        return averaged_params

    def median_average(self, parameters):

        num_layers = len(parameters[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(parameters)):
                a.append(parameters[j][i].flatten())
            aggregated_models.append(np.reshape(np.median(a, axis=0), newshape=parameters[0][i].shape))

        return aggregated_models

    def krum_average(self, k, parameters):
        # krum: return the parameter which has the lowest score defined as the sum of distance to its closest k vectors
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

        selected_idx = np.argsort(score)[0]

        return parameters[selected_idx]

    def mkrum_average(self, k, m, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  # the distance including itself, so k+1 not k

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)

        if self.dataset == 'shakespeare':
            for c in self.clients:
                c.train_data['y'], c.train_data['x'] = process_y(c.train_data['y']), process_x(c.train_data['x'])
                c.test_data['y'], c.test_data['x'] = process_y(c.test_data['y']), process_x(c.test_data['x'])

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                elif self.dataset == 'shakespeare':
                    c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                elif self.dataset == "vehicle":
                    c.train_data['y'] = c.train_data['y'] * -1
                elif self.dataset == "fmnist":
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            if self.dataset == 'celeba':
                # due to a different data storage format
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)


        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    tmp_models.append(self.local_models[idx])

                num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
                avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
                num_test, num_correct_test, _ = self.test(tmp_models)
                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            csolns = []
            losses = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for _ in range(self.local_iters):
                    data_batch = next(batches[c])
                    loss, w_global_idx = self.local_optimizer.run(
                        c, self.client_model, self.local_models[idx], 
                        self.global_model, data_batch, w_global_idx
                    )
                    losses.append(loss)

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]


                # send the malicious updates
                if idx in corrupt_id:
                    if self.boosting:
                        # scale malicious updates
                        diff = [self.clients_per_round * u for u in diff]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                if self.q == 0:
                    csolns.append(diff)
                else:
                    csolns.append((np.exp(self.q * loss), diff))

            agg_params = {
                'expected_num_benign': self.clients_per_round - int(self.clients_per_round * self.num_corrupted / len(self.clients)),
                'gradient_clipping': self.gradient_clipping
            }
            avg_updates = self.global_aggregator.run(csolns, agg_params)
            # if self.q != 0:
            #     avg_updates = self.aggregate(csolns)
            # else:
            #     if self.gradient_clipping:
            #         csolns = l2_clip(csolns)

            #     expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

            #     if self.median:
            #         avg_updates = self.median_average(csolns)
            #     elif self.k_norm:
            #         avg_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
            #     elif self.krum:
            #         avg_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
            #     elif self.mkrum:
            #         m = self.clients_per_round - expected_num_mali
            #         avg_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
            #     else:
            #         avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]

    def set_global_aggregator(self, aggregator):
        self.global_aggregator = aggregator
    
    def set_local_optimizer(self, optimizer):
        self.local_optimizer = optimizer