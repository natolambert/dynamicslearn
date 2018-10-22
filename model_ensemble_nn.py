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

        # cross_val_err_test = []
        # cross_val_err_train = []



        # iterate through the validation sets
        for train_index, test_index in kf.split(X):

        for (i, net) in enumerate(self.networks):

            dataset_cust_ind = kf.split(X).__getitem__(i)
            dataset_cust = dataset[dataset_cust_ind]
            
            # initializations that normally occur outside of loop
            # net.init_weights_orth()
            net.init_loss_fnc(dataset_cust[2],l_mean = 1,l_cov = 1) # data for std,

            # train
            acctest, acctrain = net.train_cust(dataset_cust, train_params)
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


    def save_model(self, filepath):
        torch.save(self, filepath)                  # full model state
