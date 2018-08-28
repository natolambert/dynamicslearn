# Implementation of the loss function from the following paper: https://arxiv.org/abs/1805.12114

import torch
import numpy as np
import math
import torch.nn as nn
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

    def __init__(self, idx=[0,1,2,3,4,5]):
        super(PNNLoss_Gaussian,self).__init__()

        self.idx= idx
        self.initialized_maxmin_logvar = True
        # Scalars are proportional to the variance to the loaded prediction data
        self.scalers    = torch.tensor([2.81690141, 2.81690141, 1.0, 0.02749491, 0.02615976, 0.00791358])

        # NOTE below has been moved into the generalNN object
        # if self.initialized_maxmin_logvar:
          #self.max_logvar = torch.tensor([0.4654, 1.7603, 0.635, 1.0709, 1.7087])
          #self.min_logvar = torch.tensor([-1.1057, -0.0531, -0.7361, -0.8476, -0.0607])  These minmax and scalers for pink long clean
          #self.scalers    = torch.tensor([4.2254, 4.2254, 1.0, 0.1214, 0.1509])
          # self.scalers    = torch.tensor([2.81690141, 2.81690141, 1.0, 0.02749491, 0.02615976, 0.00791358]) # scalers for pink_long_hover_clean
          # self.max_logvar = torch.tensor([18.5008, 16.2667, 19.6454, 21.7391, 26.5029, 20.1915]) # logvar for x,y,z,p,r,y from pink_long_hover_clean
          # self.min_logvar = torch.tensor([-13.0978, -10.3656, -18.3514, -14.2652, -13.6966, -8.3959]) # logvar for x,y,z,p,r,y from pink_long_hover_clean

          # New data: '_logged_data/pink-cf1/2018_08_22_cf1_hover_'
           # scalers is the variance in the delta state values
          # self.max_logvar = torch.tensor([17.51944,	11.397427,	11.536588,	8.047502,	14.243026,	7.0879607]) # logvar for x,y,z,p,r,y from pink_long_hover_clean
          # self.min_logvar = torch.tensor([-14.567484,	-9.694572,	-10.910203,	-12.0458765,	-9.731995,	-9.907068]) # logvar for x,y,z,p,r,y from pink_long_hover_clean
          # self.max_logvar = torch.nn.Parameter(torch.tensor(np.ones([1, len(self.idx)]),dtype=torch.float, requires_grad=True))
          # self.min_logvar = torch.nn.Parameter(torch.tensor(-np.ones([1, len(self.idx)]),dtype=torch.float, requires_grad=True))

    def def_datacaler(self, scalers):
        # Loads logvars for data if not initialized above HARD CODED
        if not (torch.is_tensor(scalers)):
            raise ValueError("Attempted to set a non tensor variable in the loss function")

        # sets values
        self.scalers    = scalers # scalers for pink_long_hover_clean

    def get_datascaler(self):
        return self.scalers

    def softplus_raw(self,input):
        # Performs the elementwise softplus on the input
        # softplus(x) = 1/B * log(1+exp(B*x))
        B = torch.tensor(1,dtype=torch.float)
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
        mean = output[:,:d]
        logvar = output[:,d:]

        lambda_cov = 3 # scaling the log(cov()) term in loss function
        lambda_mean = 1

        # Caps max and min log to avoid NaNs
        # OLD depreciated implementation commented
        # logvar = self.min_logvar + torch.tensor(nn.Softplus(logvar - self.min_logvar))#tmp_logvar
        # logvar = self.max_logvar - self.softplus_raw(self.max_logvar - logvar)
        # logvar = self.min_logvar + self.softplus_raw(logvar - self.min_logvar)
        logvar = max_logvar - self.softplus_raw(max_logvar - logvar)
        logvar = min_logvar + self.softplus_raw(logvar - min_logvar)

        # Computes loss
        var = torch.exp(logvar)
        b_s = mean.size()[0]    # batch size

        eps = 0 #1e-9              # Add to variance to avoid 1/0

        A = mean - target.expand_as(mean)
        A.mul_(self.scalers)
        B = torch.div(mean - target.expand_as(mean), var.add(eps))
        B.mul_(self.scalers)
        loss = lambda_mean*sum(torch.bmm(A.view(b_s, 1, -1), B.view(b_s, -1, 1)).reshape(-1,1)+lambda_cov*torch.log(torch.abs(torch.prod(var.add(eps),1)).reshape(-1,1)))
        return loss

        '''
        https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
        def mse_loss(input, target, size_average=True, reduce=True):
        """mse_loss(input, target, size_average=True, reduce=True) -> Tensor
        Measures the element-wise mean squared error.
        See :class:`~torch.nn.MSELoss` for details.
        """
        return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss,
                               input, target, size_average, reduce)

        '''
