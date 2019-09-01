'''Inspiration and help from: Nathan Lambert'''

# Import project files
# from utils import *
from utils.data import *
from utils.nn import *

# Import External Packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import pickle

# torch packages
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
#from _activation_swish import Swish
#from model_split_nn import SplitModel
# from model_split_nn_v2 import SplitModel2
import matplotlib.pyplot as plt
from collections import OrderedDict

# neural nets
#from model_split_nn import SplitModel
# from _activation_swish import Swish
# from utils_nn import *


class GeneralNN(nn.Module):
    def __init__(self, nn_params):

        super(GeneralNN, self).__init__()
        # Store the parameters:
        self.prob = nn_params['bayesian_flag']
        self.hidden_w = nn_params['hid_width']
        self.depth = nn_params['hid_depth']

        self.n_in_input = nn_params['du']
        self.n_in_state = nn_params['dx']
        self.n_in = self.n_in_input + self.n_in_state
        self.n_out = nn_params['dt']

        self.activation = nn_params['activation']
        self.d = nn_params['dropout']
        self.split_flag = nn_params['split_flag']

        self.epsilon = nn_params['epsilon']
        self.E = 0

        # Can store with a helper function for when re-loading and figuring out what was trained on
        self.state_list = []
        self.input_list = []
        self.change_state_list = []

        self.scalarX = StandardScaler()# MinMaxScaler(feature_range=(-1,1))#StandardScaler()# RobustScaler()
        self.scalarU = MinMaxScaler(feature_range=(-1,1))
        self.scalardX = MinMaxScaler(feature_range=(-1,1)) #StandardScaler() #MinMaxScaler(feature_range=(-1,1))#StandardScaler() # RobustScaler(quantile_range=(25.0, 90.0))

        # Sets loss function
        if self.prob:
            # INIT max/minlogvar if PNN
            self.max_logvar = torch.nn.Parameter(torch.tensor(1*np.ones([1, self.n_out]),dtype=torch.float, requires_grad=True))
            self.min_logvar = torch.nn.Parameter(torch.tensor(-1*np.ones([1, self.n_out]),dtype=torch.float, requires_grad=True))
            self.loss_fnc = PNNLoss_Gaussian()
            self.n_out *= 2
        else:
            self.loss_fnc = nn.MSELoss()

        layers = []
        layers.append(('dynm_input_lin', nn.Linear(
                self.n_in, self.hidden_w)))       # input layer
        layers.append(('dynm_input_act', self.activation))
        layers.append(('dynm_input_dropout', nn.Dropout(p = self.d)))
        for d in range(self.depth):
            layers.append(('dynm_lin_'+str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_'+str(d), self.activation))
            layers.append(('dynm_dropout_' + str(d), nn.Dropout(p = self.d)))


        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.n_out)))
        self.features = nn.Sequential(OrderedDict([*layers]))

    def init_weights_orth(self):
        # inits the NN with orthogonal weights
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)

        self.features.apply(init_weights)

    def forward(self, x):
        """
        Standard forward function necessary if extending nn.Module.
        """

        x = self.features(x)
        #pick out the variances, sigmoid, then apply scalar mult so that they aren't too high!
        if not self.prob:
            return x
        cutoff = int(self.n_out/2)
        var = .01/(1 + (x.narrow(1, cutoff, cutoff)).exp())
        x = torch.cat((x.narrow(1,0,cutoff), var), 1 )
        return x

    def preprocess(self, dataset):# X, U):
        """
        Preprocess X and U for passing into the neural network. For simplicity, takes in X and U as they are output from generate data, but only passed the dimensions we want to prepare for real testing. This removes a lot of potential questions that were bugging me in the general implementation. Will do the cosine and sin conversions externally.
        """
        # Already done is the transformation from
        # [yaw, pitch, roll, x_ddot, y_ddot, z_ddot]  to
        # [sin(yaw), sin(pitch), sin(roll), cos(pitch), cos(yaw),  cos(roll), x_ddot, y_ddot, z_ddot]
        # dX = np.array([utils_data.states2delta(val) for val in X])
        if len(dataset) == 3:
            X = dataset[0]
            U = dataset[1]
            dX = dataset[2]
        else:
            raise ValueError("Improper data shape for training")

        self.scalarX.fit(X)
        self.scalarU.fit(U)
        self.scalardX.fit(dX)

        # Stores the fit as tensors for PIPPS policy propogation through network
        if True:
            # U is a minmax scalar from -1 to 1
            # X is a standard scalar, mean 0, sigma 1
            self.scalarU_tensors_d_min = torch.FloatTensor(self.scalarU.data_max_)
            self.scalarU_tensors_d_max = torch.FloatTensor(self.scalarU.data_min_)
            self.scalarU_tensors_d_range = torch.FloatTensor(self.scalarU.data_range_)
            self.scalarU_tensors_f_range = torch.FloatTensor([-1,1])

            self.scalarX_tensors_mean = torch.FloatTensor(self.scalarX.mean_)
            self.scalarX_tensors_var = torch.FloatTensor(self.scalarX.var_)

            self.scalardX_tensors_d_min = torch.FloatTensor(
                self.scalardX.data_min_)
            self.scalardX_tensors_scale = torch.FloatTensor(
                self.scalardX.scale_)
        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(X)
        normU = self.scalarU.transform(U)
        normdX = self.scalardX.transform(dX)


        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(normdX)

        return list(zip(inputs, outputs))



    def postprocess(self, dX):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
        """
        # de-normalize so to say
        dX = self.scalardX.inverse_transform(dX.reshape(1,-1))
        dX = dX.ravel()
        return np.array(dX)


    def train_cust(self, dataset, train_params, gradoff = False):
        """
        Train the neural network.
        if preprocess = False
            dataset is a list of tuples to train on, where the first value in the tuple is the training data (should be implemented as a torch tensor), and the second value in the tuple
            is the label/action taken
        if preprocess = True
            dataset is simply the raw output of generate data (X, U)
        Epochs is number of times to train on given training data,
        batch_size is hyperparameter dicating how large of a batch to use for training,
        optim is the optimizer to use (options are "Adam", "SGD")
        split is train/test split ratio
        """
        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        optim = train_params['optim']
        split = train_params['split']
        lr = train_params['lr']
        lr_step_eps = train_params['lr_schedule'][0]
        lr_step_ratio = train_params['lr_schedule'][1]
        preprocess = train_params['preprocess']
        momentum = train_params['momentum']
        self.train()
        if preprocess:
            dataset = self.preprocess(dataset)#[0], dataset[1])
            # print('Shape of dataset is:', len(dataset))

        if self.prob:
            loss_fn = PNNLoss_Gaussian(idx=np.arange(0,self.n_out/2,1))
            self.test_loss_fnc = loss_fn
            # self.test_loss_fnc = MSELoss()
        else:
            loss_fn = MSELoss()

        # makes sure loss fnc is correct
        if loss_fn == PNNLoss_Gaussian() and not self.prob:
            raise ValueError('Check NN settings. Training a deterministic net with pnnLoss. Pass MSELoss() to train()')

        trainLoader = DataLoader(dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)

        # Papers seem to say ADAM works better
        if(optim=="Adam"):
            optimizer = torch.optim.Adam(super(GeneralNN, self).parameters(), lr=lr)
        elif (optim == 'SGD'):
            optimizer = torch.optim.SGD(super(GeneralNN, self).parameters(), lr = lr, momentum = momentum)
        else:
            raise ValueError(optim + " is not a valid optimizer type")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_eps, gamma=lr_step_ratio)

        testloss, trainloss = self._optimize(self.loss_fnc, optimizer, split, scheduler, epochs, batch_size, dataset)
        return testloss, trainloss

    def predict(self, X, U):
        """
        Given a state X and input U, predict the next state. This function is used when simulating, so it does all pre and post processing for the neural net
        """
        dx = len(X)
        self.eval()

        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))


        input = torch.Tensor(np.concatenate((normX, normU), axis=1))
        NNout = self.forward(input).data[0]
        # If probablistic only takes the first half of the outputs for predictions
        '''Implement more generality. This returns means and variance if it is probabilistic but only returns mean if deterministic.'''
        if self.prob:
            mean = self.postprocess(NNout[:int(self.n_out/2)]).ravel()
            var = NNout[int(self.n_out/2):]
            return mean,var
        else:
            NNout = self.postprocess(NNout).ravel()

        return NNout

    def _optimize(self, loss_fn, optim, split, scheduler, epochs, batch_size, dataset, gradoff=False): #trainLoader, testLoader):
        errors = []
        error_train = []
        split = split

        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)
        trainLoader = DataLoader(dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            scheduler.step()
            avg_loss = torch.zeros(1)
            num_batches = len(trainLoader)/batch_size
            for i, (input, target) in enumerate(trainLoader):
                optim.zero_grad()
                output = self.forward(input).float()
                if self.prob:
                    loss = loss_fn(output, target, self.max_logvar, self.min_logvar)
                else:
                    loss = loss_fn(output, target)
                # add small loss term on the max and min logvariance if probablistic network
                # note, adding this term will backprob the values properly
                # lambda_logvar = torch.FloatTensor([.01])
                lambda_logvar = .01
                if self.prob:
                    # print(loss)
                    # print(lambda_logvar * torch.sum((self.max_logvar)))
                    loss += lambda_logvar * torch.sum((self.max_logvar)) - lambda_logvar * torch.sum((self.min_logvar))
                if loss.data.numpy() == loss.data.numpy():
                    # print(self.max_logvar, self.min_logvar)
                    if not gradoff:
                        loss.backward()                               # backpropagate from the loss to fill the gradient buffers
                        optim.step()                                  # do a gradient descent step
                    # print('tain: ', loss.item())
                else:
                    '''TODO: implement a reset in neural network state dicts and variables so that when we have NaN, just start over and retrain the network on the same parameters'''
                    print("loss is NaN")                       # This is helpful: it'll catch that when it happens,
                    print("Stopping training at ", epoch, " epochs.")
                    # print("Output: ", output, "\nInput: ", input, "\nLoss: ", loss)
                    errors.append(np.nan)
                    error_train.append(np.nan)
                    return errors, error_train                 # and give the output and input that made the loss NaN
                avg_loss += (loss.item()/(len(trainLoader)*batch_size))
                '''Editted to return list of losses every epoch'''# update the overall average loss with this batch's loss

            self.eval()
            test_error = torch.zeros(1)
            for i, (input, target) in enumerate(testLoader):

                output = self.forward(input)
                # means = output[:,:9]
                if self.prob:
                    loss = loss_fn(output, target, self.max_logvar, self.min_logvar)                # compute the loss
                else:
                    loss = loss_fn(output, target)
                test_error += loss.item()/(len(testLoader)*batch_size)
            test_error = test_error
            #print("Look for increase in loss here:", avg_loss[0])
            #if (epoch % 1 == 0): print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(avg_loss.data[0]), "test loss=", "{:.6f}".format(test_error.data[0]))
            error_train.append(avg_loss[0].tolist())
            errors.append(test_error[0].tolist())
        return errors, error_train

    def save_model(self, filepath):
        torch.save(self, filepath)
        print("saved")                 # full model state
        # For load, use torch.load()
