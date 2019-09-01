'''Code from: Nathan Lambert'''

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

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
        self.scalers = torch.tensor([1, 1, 1, 1, 1, 1, 1,  1, 1])

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
        return (torch.log(1+torch.exp(input.mul_(B)))).div_(B)

    def forward(self, output, target, max_logvar, min_logvar):
        '''
        output is a vector of length 2d
        mean is a vector of length d, which is the first set of outputs of the PNN
        var is a vector of variances for each of the respective means
        target is a vector of the target values for each of the mean
        '''

        # Initializes parameterss
        d2 = output.size()[1]
        d = torch.tensor(d2/2, dtype=torch.int32)
        mean = output[:, :d]
        logvar = output[:, d:]

        # Caps max and min log to avoid NaNs
        logvar = max_logvar - self.softplus_raw(max_logvar - logvar)
        logvar = min_logvar + self.softplus_raw(logvar - min_logvar)

        # Computes loss
        var = torch.exp(logvar)
        b_s = mean.size()[0]    # batch size

        eps = 0              # Add to variance to avoid 1/0

        A = mean - target.expand_as(mean)
        A.mul_(self.scalers)
        B = torch.div(mean - target.expand_as(mean), var.add(eps))
        # B.mul_(self.scalers)
        loss = torch.sum(self.lambda_mean*torch.bmm(A.view(b_s, 1, -1), B.view(b_s, -1, 1)).reshape(-1, 1)+self.lambda_cov*torch.log(torch.abs(torch.prod(var.add(eps),1)).reshape(-1,1)))
        return loss
