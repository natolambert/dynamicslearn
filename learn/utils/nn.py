import copy
import math
import numpy as np
import torch
import torch.nn as nn
import hydra
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer


class ModelDataHandler:
    def __init__(self, **params):
        # needs to have three types of scikitlearn preprocessing objects in the
        self.scalarX = hydra.utils.instantiate(params['X'])  # ['X'].type(**params['X'].params)
        self.scalarU = hydra.utils.instantiate(params['U'])  # ['X'].type(**params['X'].params)
        self.scalardX = hydra.utils.instantiate(params['dX'])  # ['X'].type(**params['X'].params)
        # self.scalarU = params['U'].type(**params['U'].params)
        # self.scalardX = params['dX'].type(**params['dX'].params)

        self.sine_transform = params['sine_expand']
        self.fit = False

    def forward(self, X, U):
        if not self.fit:
            raise ValueError("Fit the normalization before trying to use it")
        if len(np.shape(X)) > 1:
            l = np.shape(X)[0]
        else:
            l = 1
        # normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(l, -1))
        normU = self.scalarU.transform(U.reshape(l, -1))
        return normX, normU

    def preprocess(self, dataset, ret_data=True):
        # Get parts
        if len(dataset) == 3:
            X = dataset[0]
            U = dataset[1]
            dX = dataset[2]
        else:
            raise ValueError("Improper data shape for training")

        # Transform if needed
        if len(self.sine_transform) > 0:
            raise NotImplementedError("Not Done Yet")

        # fit
        if ret_data:
            self.scalarX.fit(X)
            self.scalarU.fit(U)
            self.scalardX.fit(dX)

            # Normalizing to zero mean and unit variance
            normX = self.scalarX.transform(X)
            normU = self.scalarU.transform(U)
            normdX = self.scalardX.transform(dX)

            inputs = np.concatenate((normX, normU), axis=1)
            outputs = normdX

            self.fit = True
            return inputs, outputs
        else:
            self.scalarX.fit(X)
            self.scalarU.fit(U)
            self.scalardX.fit(dX)

            self.fit = True
            return self.scalarX, self.scalarU, self.scalardX

    def postprocess(self, dX):
        if len(np.shape(dX)) > 1:
            l = np.shape(dX)[0]
        else:
            l = 1
        dX = self.scalardX.inverse_transform(dX.reshape(l, -1)).squeeze()
        return dX


class Swish(nn.Module):
    def __init__(self, B=1.0):
        super(Swish, self).__init__()
        self.B = B

    def forward(self, x):
        Bx = x.mul(self.B)
        omega = (1 + Bx.exp()) ** (-1)
        return x.mul(omega)


class PNNLoss_Gaussian(torch.nn.Module):
    '''
    Here is a brief aside on why we want and will use this loss. Essentially, we will incorporate this loss function to include a probablistic nature to the dynamics learning nueral nets. The output of the Probablistic Nueral Net (PNN) or Bayesian Neural Net (BNN) will be both a mean for each trained variable and an associated variance. This loss function will take the mean (u), variance (sig), AND the true trained value (s) to compare against the mean. Stacked variances form Cov matrix

    loss_gaussian = sum_{data} (u - s)^T Cov^-1 (u-s) + log Det(Cov)

    Need to add code like this to the implementation:
         To bound the variance output for a probabilistic network to be between the upper and lower bounds found during training the network on the training data, we used the following code with automatic differentiation:

         logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
         logvar = min_logvar + tf.nn.softplus(logvar - min_logvar)
         var = tf.exp(logvar)

         with a small regularization penalty on term on max_logvar so that it does not grow beyond the training distribution’s maximum output variance, and on the negative of min_logvar so that it does not drop below the training distribution’s minimum output variance.
    '''

    def __init__(self, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        super(PNNLoss_Gaussian, self).__init__()

        self.idx = idx
        self.initialized_maxmin_logvar = True
        # Scalars are proportional to the variance to the loaded prediction data
        # self.scalers    = torch.tensor([2.81690141, 2.81690141, 1.0, 0.02749491, 0.02615976, 0.00791358])
        self.scalers = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])

        # weight the parts of loss
        self.lambda_cov = 1  # scaling the log(cov()) term in loss function
        self.lambda_mean = 1

    def set_lambdas(self, l_mean, l_cov):
        # sets the weights of the loss function
        self.lambda_cov = l_mean
        self.lambda_mean = l_cov

    def get_datascaler(self):
        return self.scalers

    def softplus_raw(self, input):
        # Performs the elementwise softplus on the input
        # softplus(x) = 1/B * log(1+exp(B*x))
        B = torch.tensor(1, dtype=torch.float)
        return (torch.log(1 + torch.exp(input.mul_(B)))).div_(B)

    def forward(self, output, target, max_logvar, min_logvar):
        '''
        output is a vector of length 2d
        mean is a vector of length d, which is the first set of outputs of the PNN
        var is a vector of variances for each of the respective means
        target is a vector of the target values for each of the mean
        '''

        # Initializes parameterss
        d2 = output.size()[1]
        d = torch.tensor(d2 / 2, dtype=torch.int32)
        mean = output[:, :d]
        logvar = output[:, d:]

        # Caps max and min log to avoid NaNs
        logvar = max_logvar - self.softplus_raw(max_logvar - logvar)
        logvar = min_logvar + self.softplus_raw(logvar - min_logvar)

        # Computes loss
        var = torch.exp(logvar)
        b_s = mean.size()[0]  # batch size

        eps = 0  # Add to variance to avoid 1/0

        # A = mean - target.expand_as(mean)
        # A.mul_(self.scalers)
        # B = torch.div(mean - target.expand_as(mean), var.add(eps))
        # # B.mul_(self.scalers)
        # loss = torch.sum(self.lambda_mean * torch.bmm(A.view(b_s, 1, -1), B.view(b_s, -1, 1)).reshape(-1,1) + self.lambda_cov * torch.log(torch.abs(torch.prod(var.add(eps), 1)).reshape(-1, 1)))

        A = self.lambda_mean * torch.sum(((mean - target.expand_as(mean)) ** 2) / var, 1)

        B = self.lambda_cov * torch.log(torch.abs(torch.prod(var.add(eps), 1)))

        loss = torch.sum(A + B)

        return loss


def predict_nn(model, x, u, indexlist):
    '''
    special, generalized predict function for the general nn class in construction.
    x, u are vectors of current state and input to get next state or change in state
    indexlist is is an ordered index list for which state variable the indices of the input to the NN correspond to. Assumes states come before any u
    '''

    # Makes prediction for either prediction mode. Handles the need to only pass certain states
    prediction = np.copy(x)
    pred = model.predict(x, u)
    for i, idx in enumerate(indexlist):
        # print('x_nn = ', x[idx], 'predicted', pred)
        prediction[idx] = x[idx] + pred[i]

    return prediction


def predict_nn_v2(model, x, u, targetlist=[]):
    '''
    special, generalized predict function for the general nn class in construction.
    x, u are vectors of current state and input to get next state or change in state
    Training list tells whether or not each input is a raw state or a change in state
    '''
    # important this list is in order
    if targetlist == []:
        _, _, targetlist = model.get_training_lists()

    # generate labels as to whether or not it true state or delta
    # true = delta
    lab = [t[:2] == 'd_' for t in targetlist]

    # Makes prediction for either prediction mode. Handles the need to only pass certain states
    prediction = np.zeros(9)
    pred = model.predict(x, u)
    # print("Pred in standard way")
    # print(pred)
    # print("~~~")
    for i, l in enumerate(lab):
        if l:
            prediction[i] = x[i] + pred[i]
        else:
            prediction[i] = pred[i]

    return prediction


# LEGACY CODE BELOW
# module that sends the [0, nn_in - split] inputs through one network and the [split, nn_in] inputs through another
class SplitModel(nn.Module):
    def __init__(self, nn_in, nn_out, width, prob=True, activation='swish', dropout=0.02):
        super(SplitModel, self).__init__()

        self.nn_in = nn_in
        self.nn_out = nn_out
        self.prob = prob
        self.dropout = dropout
        self.width = width

        if activation.lower() == 'swish':
            self.activation = Swish(B=1.0)
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()

        # Common input layer:
        self.main = nn.Sequential(nn.Linear(self.nn_in, self.width),
                                  copy.deepcopy(self.activation),
                                  nn.Dropout(p=self.dropout))

        # Angular accel model
        self.angular = nn.Sequential(nn.Linear(self.width, self.width),
                                     copy.deepcopy(self.activation),
                                     nn.Dropout(p=self.dropout),
                                     nn.Linear(self.width, self.width),
                                     copy.deepcopy(self.activation),
                                     nn.Dropout(p=self.dropout),
                                     nn.Linear(self.width, int(nn_out / 3)))

        # Euler Angles Model accel model
        self.euler = nn.Sequential(nn.Linear(self.width, self.width),
                                   copy.deepcopy(self.activation),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.width, self.width),
                                   copy.deepcopy(self.activation),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.width, int(nn_out / 3)))

        # Linear accel model
        self.linear = nn.Sequential(nn.Linear(self.width, self.width),
                                    copy.deepcopy(self.activation),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.width, self.width),
                                    copy.deepcopy(self.activation),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.width, int(nn_out / 3)))

    def forward(self, x):
        x = self.main(x)
        angular = self.angular(x)
        euler = self.euler(x)
        linear = self.linear(x)

        if self.prob:
            means = torch.cat((angular[:, :3], euler[:, :3], linear[:, :3]), 1)
            variances = torch.cat((angular[:, 3:], euler[:, 3:], linear[:, 3:]), 1)
            x = torch.cat((means, variances), 1)
        else:
            x = torch.cat((angular, euler, linear), 1)

        return x
