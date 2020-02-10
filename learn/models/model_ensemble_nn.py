# Our infrastucture files
# from utils import *

# data packages
import numpy as np
# Torch Packages
import torch
import torch.nn as nn
# More NN such
from sklearn.model_selection import KFold  # for dataset

# neural nets
from learn.models.model_general_nn import GeneralNN
import hydra


class EnsembleNN(nn.Module):
    '''
    This file is in the works for an object to easily create an ensemble model. These
      models will be used heavily for offline bootstrapping of policies and controllers.
    '''

    def __init__(self, **nn_params):
        super(EnsembleNN, self).__init__()
        self.E = nn_params['training']['E']  # number of networks to use in each ensemble
        self.prob = nn_params['training']['probl']
        self.dx = nn_params['dx']

        self.hist = nn_params['history']
        ex_in = len(nn_params['extra_inputs']) if nn_params['extra_inputs'] is not None else 0
        self.n_in_input = nn_params['du'] * (self.hist + 1) + ex_in
        self.n_in_state = nn_params['dx'] * (self.hist + 1)
        self.n_in = self.n_in_input + self.n_in_state
        self.n_out = nn_params['dt']
        self.p_out = nn_params['dt']
        if self.prob:
            self.n_out *= 2

        # create networks
        self.networks = []
        for i in range(self.E):
            # self.networks.append(hydra.utils.instantiate(nn_params))
            self.networks.append(GeneralNN(**nn_params))

        # Can store with a helper function for when re-loading and figuring out what was trained on
        self.state_list = []
        self.input_list = []
        self.change_state_list = []

    def train_cust(self, dataset, train_params):
        '''
        To train the enemble model simply train each subnetwork on the same data
        Will return the test and train accuracy in lists of 1d arrays
        '''

        acctest_l = []
        acctrain_l = []

        # setup cross validation-ish datasets for training ensemble
        kf = KFold(n_splits=self.E)
        kf.get_n_splits(dataset)

        # iterate through the validation sets
        for (i, net), (train_idx, test_idx) in zip(enumerate(self.networks), kf.split(dataset[0])):
            # only train on training data to ensure diversity
            X_cust = dataset[0][train_idx, :]
            U_cust = dataset[1][train_idx, :]
            dX_cust = dataset[2][train_idx, :]

            # initializations that normally occur outside of loop
            # net.init_weights_orth()
            if self.prob: net.init_loss_fnc(dX_cust, l_mean=1, l_cov=1)  # data for std,

            # train
            acctest, acctrain = net.train_cust((X_cust, U_cust, dX_cust), train_params)
            acctest_l.append(acctest)
            acctrain_l.append(acctrain)

        return np.transpose(np.array(acctest_l)), np.transpose(np.array(acctrain_l))

    def predict(self, X, U, ret_var=False):
        prediction = np.zeros(int(self.n_out/2)) if self.prob else np.zeros(self.n_out)
        # vars = torch.zeros(())
        for net in self.networks:
            if ret_var:
                # raise NotImplementedError("Need to handle Variance Returns")
                prediction += (1 / self.E) * net.predict(X, U)
            else:
                prediction += (1 / self.E) * net.predict(X, U)

        return prediction, torch.Tensor(1)

    def distribution(self, state, action):
        """
        Takes in a state, action pair and returns a probability distribution for each state composed of mean and variances for each state:
        - Needs to normalize the state and the action
        - Needs to scale the state and action distrubtions on the back end to match up.
        - Should be a combo of forward and pre/post processing
        """
        dx = self.dx
        means = torch.zeros((dx, 1))
        var = torch.zeros((dx, 1))
        for net in self.networks:
            means_e, var_e = net.distribution(state, action)
            means = means + means_e.reshape(-1, 1) / self.E
            var = var + var_e.reshape(-1, 1) / self.E

        return means.squeeze(), var.squeeze()

    def getNormScalers(self):
        # all the data passed in is the same, so the scalers are identical
        return self.networks[0].getNormScalers()

    def store_training_lists(self, state_list=[], input_list=[], change_state_list=[]):
        # stores the column labels of the generated dataframe used to train this network
        self.state_list = state_list
        self.input_list = input_list
        self.change_state_list = change_state_list

    def get_training_lists(self):
        # return the training lists for inspection
        return self.state_list, self.input_list, self.change_state_list

    def save_model(self, filepath):
        torch.save(self, filepath)  # full model state
