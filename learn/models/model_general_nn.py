# Import project files
# from utils import *
from ..utils.data import *
from ..utils.nn import *
from .model import DynamicsModel

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

import matplotlib.pyplot as plt
from collections import OrderedDict


class GeneralNN(nn.Module):
    def __init__(self, **nn_params):
        super(GeneralNN, self).__init__()
        """
        Simpler implementation of my other neural net class. After parameter tuning, now just keep the structure and change it if needed. Note that the data passed into this network is only that which is used.
        """
        # Store the parameters:
        self.prob = nn_params['training']['probl']
        self.hidden_w = nn_params['training']['hid_width']
        self.depth = nn_params['training']['hid_depth']

        self.hist = nn_params['history']
        self.n_in_input = nn_params['du']*self.hist
        self.n_in_state = nn_params['dx']*self.hist
        self.n_in = self.n_in_input + self.n_in_state
        self.n_out = nn_params['dt']

        self.activation = Swish()  # hydra.utils.instantiate(nn_params['training']['activ'])
        self.d = nn_params['training']['dropout']
        self.split_flag = nn_params['training']['split']

        self.E = 0  # clarify that these models are not ensembles

        # Can store with a helper function for when re-loading and figuring out what was trained on
        self.state_list = []
        self.input_list = []
        self.change_state_list = []

        self.scalarX = StandardScaler()  # MinMaxScaler(feature_range=(-1,1))#StandardScaler()# RobustScaler()
        self.scalarU = MinMaxScaler(feature_range=(-1, 1))
        self.scalardX = MinMaxScaler(feature_range=(-1, 1))

        self.init_training = False
        # Sets loss function
        if self.prob:
            # INIT max/minlogvar if PNN
            self.max_logvar = torch.nn.Parameter(
                torch.tensor(1 * np.ones([1, self.n_out]), dtype=torch.float, requires_grad=True))
            self.min_logvar = torch.nn.Parameter(
                torch.tensor(-1 * np.ones([1, self.n_out]), dtype=torch.float, requires_grad=True))
            self.loss_fnc = PNNLoss_Gaussian()
            self.n_out *= 2
        else:
            self.loss_fnc = nn.MSELoss()

        # create object nicely
        layers = []
        layers.append(('dynm_input_lin', nn.Linear(
            self.n_in, self.hidden_w)))  # input layer
        layers.append(('dynm_input_act', self.activation))
        # layers.append(nn.Dropout(p=self.d))
        for d in range(self.depth):
            # add modules
            # input layer
            layers.append(
                ('dynm_lin_' + str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_' + str(d), self.activation))
            # layers.append(nn.Dropout(p=self.d))

        # output layer
        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.n_out)))
        # print(*layers)
        self.features = nn.Sequential(OrderedDict([*layers]))

    def init_weights_orth(self):
        # inits the NN with orthogonal weights
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)

        self.features.apply(init_weights)

    def init_loss_fnc(self, targets, l_mean=1, l_cov=1):
        if not self.prob:
            raise ValueError('Attempted to set minmaxlog_var of non bayesian network')

        # updates targets of the loss_fnc
        self.loss_fnc.scalers = torch.Tensor(np.std(targets, axis=0))

        self.loss_fnc.set_lambdas(l_mean, l_cov)

    def getNormScalers(self):
        return self.scalarX, self.scalarU, self.scalardX

    def store_training_lists(self, state_list=[], input_list=[], change_state_list=[]):
        # stores the column labels of the generated dataframe used to train this network
        self.state_list = state_list
        self.input_list = input_list
        self.change_state_list = change_state_list

    def get_training_lists(self):
        # return the training lists for inspection
        return self.state_list, self.input_list, self.change_state_list

    def forward(self, x):
        """
        Standard forward function necessary if extending nn.Module.
        """

        x = self.features(x)
        return x  # x.view(x.size(0), -1)

    def distribution(self, state, action):
        """
        Takes in a state, action pair and returns a probability distribution for each state composed of mean and variances for each state:
        - Needs to normalize the state and the action
        - Needs to scale the state and action distrubtions on the back end to match up.
        - Should be a combo of forward and pre/post processing
        """
        # NORMALIZE ======================================================
        # normalize states because the action gradients affect future states
        normX_T = (state - self.scalarX_tensors_mean) / torch.sqrt(self.scalarX_tensors_var)
        # need to normalize the action with tensors to keep the gradient path
        U_std = (action - self.scalarU_tensors_d_min) / \
                (self.scalarU_tensors_d_max - self.scalarU_tensors_d_min)
        normU_T = U_std * \
                  (self.scalarU_tensors_f_range[0] - self.scalarU_tensors_f_range[-1]) + \
                  self.scalarU_tensors_f_range[-1]
        # normU = self.scalarU.transform(action.reshape(1, -1))

        # FORWARD ======================================================
        # print( torch.cat((normX_T, normU_T), 1).view(-1) )
        out = self.forward(
            torch.cat((normX_T, normU_T), 0)).view(-1)

        # print(out)
        l = int(len(out) / 2)
        means = out[:l]
        logvar = out[l:]

        # DE-NORMALIZE ======================================================
        # to denormalize, add 1 so 0 to 2, divide by the scale, add the min
        means = ((means + 1.) / self.scalardX_tensors_scale) + self.scalardX_tensors_d_min
        # means = means*self.scalarX_tensors_var[:l]+self.scalarX_tensors_mean[:l]
        var = torch.exp(logvar)  # because of how the loss function is created
        return means, var

    def preprocess(self, dataset):  # X, U):
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

        # Stores the fit as tensors for offline prediction, etc
        if True:
            # U is a minmax scalar from -1 to 1
            # X is a standard scalar, mean 0, sigma 1
            self.scalarU_tensors_d_min = torch.FloatTensor(self.scalarU.data_max_)
            self.scalarU_tensors_d_max = torch.FloatTensor(self.scalarU.data_min_)
            self.scalarU_tensors_d_range = torch.FloatTensor(self.scalarU.data_range_)
            self.scalarU_tensors_f_range = torch.FloatTensor([-1, 1])

            self.scalarX_tensors_mean = torch.FloatTensor(self.scalarX.mean_)
            self.scalarX_tensors_var = torch.FloatTensor(self.scalarX.var_)

            self.scalardX_tensors_d_min = torch.FloatTensor(self.scalardX.data_min_)
            self.scalardX_tensors_scale = torch.FloatTensor(self.scalardX.scale_)

        # Normalizing to zero mean and unit variance
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
        if len(np.shape(dX)) > 1:
            l = np.shape(dX)[0]
        else:
            l = 1
        dX = self.scalardX.inverse_transform(dX.reshape(l, -1)).squeeze()
        return np.array(dX)

    def train_cust(self, dataset, model_params, gradoff=False):
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
        # Handle inizializations on first call
        if self.init_training == False:
            self.init_weights_orth()
            if self.prob: self.init_loss_fnc(dataset[2], l_mean=1, l_cov=1)  # data for std,
        self.init_training = True

        train_params = model_params.optimizer
        epochs = train_params['epochs']
        batch_size = train_params['batch']
        split = train_params['split']
        lr = train_params['lr']
        lr_step_eps = train_params['lr_schedule'][0]
        lr_step_ratio = train_params['lr_schedule'][1]
        preprocess = train_params['preprocess']

        if preprocess:
            dataset = self.preprocess(dataset)  # [0], dataset[1])
            # print('Shape of dataset is:', len(dataset))

        if self.prob:
            loss_fn = PNNLoss_Gaussian(idx=np.arange(0, self.n_out / 2, 1))
            self.test_loss_fnc = loss_fn
            # self.test_loss_fnc = MSELoss()
        else:
            loss_fn = MSELoss()

        # makes sure loss fnc is correct
        if loss_fn == PNNLoss_Gaussian() and not self.prob:
            raise ValueError('Check NN settings. Training a deterministic net with pnnLoss. Pass MSELoss() to train()')

        trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=batch_size)

        # Papers seem to say ADAM works better
        optimizer = torch.optim.Adam(super(GeneralNN, self).parameters(), lr=lr)
        # optimizer = torch.optim.SGD(super(GeneralNN, self).parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6,
                                                    gamma=0.7)  # most results at .6 gamma, tried .33 when got NaN

        testloss, trainloss = self._optimize(self.loss_fnc, optimizer, split, scheduler, epochs, batch_size,
                                             dataset)  # trainLoader, testLoader)
        return testloss, trainloss

    def predict(self, X, U, ret_var=False):
        """
        Given a state X and input U, predict the change in state dX. This function is used when simulating, so it does all pre and post processing for the neural net
        """
        if len(np.shape(X)) > 1:
            l = np.shape(X)[0]
        else:
            l = 1

        normX = self.scalarX.transform(X.reshape(l, -1))
        normU = self.scalarU.transform(U.reshape(l, -1))

        input = torch.Tensor(np.concatenate((normX, normU), axis=1))

        NNout = self.forward(input).data

        # If probablistic only takes the first half of the outputs for predictions
        if self.prob:
            ret = self.postprocess(NNout[:, :int(self.n_out / 2)]).squeeze()
            if ret_var:
                return ret, NNout[:, int(self.n_out / 2):]
        else:
            ret = self.postprocess(NNout).squeeze()

        return ret

    def _optimize(self, loss_fn, optim, split, scheduler, epochs, batch_size, dataset,
                  gradoff=False):  # trainLoader, testLoader):
        errors = []
        error_train = []
        split = split

        testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=batch_size)
        trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):

            avg_loss = torch.zeros(1)
            # num_batches = len(trainLoader) / batch_size
            for i, (input, target) in enumerate(trainLoader):
                # Add noise to the batch
                if False:
                    if self.prob:
                        n_out = int(self.n_out / 2)
                    else:
                        n_out = self.n_out
                    noise_in = torch.Tensor(np.random.normal(0, .01, (input.size())), dtype=torch.float)
                    noise_targ = torch.Tensor(np.random.normal(0, .01, (target.size())), dtype=torch.float)
                    input.add_(noise_in)
                    target.add_(noise_targ)

                optim.zero_grad()  # zero the gradient buffers
                output = self.forward(input)  # compute the output
                if self.prob:
                    loss = loss_fn(output, target, self.max_logvar, self.min_logvar)  # compute the loss
                else:
                    loss = loss_fn(output, target)

                # add small loss term on the max and min logvariance if probablistic network
                # note, adding this term will backprob the values properly
                lambda_logvar = .01
                if self.prob:
                    loss += lambda_logvar * torch.sum((self.max_logvar)) - lambda_logvar * torch.sum((self.min_logvar))

                if loss.data.numpy() == loss.data.numpy():
                    if not gradoff:
                        loss.backward()  # backpropagate from the loss to fill the gradient buffers
                        optim.step()  # do a gradient descent step

                # if not loss.data.numpy() == loss.data.numpy(): # Some errors make the loss NaN. this is a problem.
                else:
                    print("loss is NaN")  # This is helpful: it'll catch that when it happens,
                    # print("Output: ", output, "\nInput: ", input, "\nLoss: ", loss)
                    errors.append(np.nan)
                    error_train.append(np.nan)
                    return errors, error_train  # and give the output and input that made the loss NaN
                avg_loss += loss.item() / (
                        len(trainLoader) * batch_size)  # update the overall average loss with this batch's loss

            test_error = torch.zeros(1)
            for i, (input, target) in enumerate(testLoader):
                output = self.forward(input)
                if self.prob:
                    loss = loss_fn(output, target, self.max_logvar, self.min_logvar)  # compute the loss
                else:
                    loss = loss_fn(output, target)

                test_error += loss.item() / (len(testLoader) * batch_size)
            test_error = test_error
            # if (epoch % 1 == 0): print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(avg_loss.data[0]),
            #                            "test loss=", "{:.6f}".format(test_error.data[0]))
            error_train.append(avg_loss.data[0].numpy())
            errors.append(test_error.data[0].numpy())
            scheduler.step()

        return errors, error_train

    def save_model(self, filepath):
        torch.save(self, filepath)
