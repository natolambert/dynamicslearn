# Our infrastucture files
from utils_data import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN, predict_nn
from model_split_nn import SplitModel
from _activation_swish import Swish

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal

# More NN such
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import pickle
from sklearn.model_selection import KFold   # for dataset


from _activation_swish import Swish
from _lossfnc_pnngaussian import PNNLoss_Gaussian

import matplotlib.pyplot as plt

# my NN object
from model_general_nn import GeneralNN


class EnsembleNN(nn.Module):
    '''
    This file is in the works for an object to easily create an ensemble model. These
      models will be used heavily for offline bootstrapping of policies and controllers.

      Some features that should be implemented:
      - Easily predict the output of the network with forward()
      - change whether or not the models are trained on separate datasets sampled with
        replacement or relying on SGD / random initializations to get different networks
        (Do I have to check random seeds for this?)

    '''
    def __init__(self, nn_params, E=5, forward_mode = ''):
        super(EnsembleNN, self).__init__()
        self.E = E              # number of networks to use in each ensemble
        self.forward_mode = ''  # TODO: implement a weighted set of predictions based on confidence

        # create networks
        self.networks = []
        for i in range(E):
            self.networks.append(GeneralNN(nn_params))

        # Can store with a helper function for when re-loading and figuring out what was trained on
        self.state_list = []
        self.input_list = []
        self.change_state_list = []

    def train_cust(self, dataset, train_params, gradoff = False):
        '''
        To train the enemble model simply train each subnetwork on the same data
        Will return the test and train accuracy in lists of 1d arrays
        '''

        acctest_l = []
        acctrain_l = []

        # setup cross validation-ish datasets for training ensemble
        kf = KFold(n_splits=self.E)
        kf.get_n_splits(dataset)

        # cross_val_err_test = []
        # cross_val_err_train = []


        if gradoff:
            err = 0
            for (i, net) in enumerate(self.networks):
                train_params = {
                    'epochs' : 1,
                    'batch_size' : 32,
                    'optim' : 'Adam',
                    'split' : 0.99,
                    'lr': .002,
                    'lr_schedule' : [30,.6],
                    'test_loss_fnc' : [],
                    'preprocess' : True,
                    'noprint' : True
                }
                acctest, acctrain = net.train_cust((dataset), train_params, gradoff = True)
                err += min(acctrain)/self.E
            return 0, err
        else:
            # iterate through the validation sets
            for (i, net), (train_idx, test_idx) in zip(enumerate(self.networks),kf.split(dataset[0])):
                # only train on training data to ensure diversity
                X_cust = dataset[0][train_idx,:]
                U_cust = dataset[1][train_idx,:]
                dX_cust = dataset[2][train_idx,:]

                # initializations that normally occur outside of loop
                # net.init_weights_orth()
                net.init_loss_fnc(dX_cust,l_mean = 1,l_cov = 1) # data for std,

                # train
                acctest, acctrain = net.train_cust((X_cust, U_cust, dX_cust), train_params)
                acctest_l.append(acctest)
                acctrain_l.append(acctrain)


        return np.transpose(np.array(acctest_l)), np.transpose(np.array(acctrain_l))

    def predict(self, X, U):
        prediction = np.zeros([9])

        for net in self.networks:
            prediction += (1/self.E)*net.predict(X,U)

        return prediction

    def getNormScalers(self):
        # all the data passed in is the same, so the scalers are identical
        return self.networks[0].getNormScalers()

    def store_training_lists(self, state_list = [], input_list = [], change_state_list = []):
        # stores the column labels of the generated dataframe used to train this network
        self.state_list = state_list
        self.input_list = input_list
        self.change_state_list = change_state_list

    def get_training_lists(self):
        # return the training lists for inspection
        return self.state_list, self.input_list, self.change_state_list


    def save_model(self, filepath):
        torch.save(self, filepath)                  # full model state
