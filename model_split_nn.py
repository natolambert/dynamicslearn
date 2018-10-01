import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from _activation_swish import Swish

# module that sends the [0, nn_in - split] inputs through one network and the [split, nn_in] inputs through another
class SplitModel(nn.Module):

  def __init__(self, nn_in, nn_out, width, prob=True, activation = 'swish', dropout = 0.02):
    super(SplitModel2, self).__init__()

    self.nn_in      = nn_in
    self.nn_out     = nn_out
    self.prob       = prob
    self.dropout    = dropout
    self.width = width

    if activation.lower() == 'swish':
      self.activation = Swish(B = 1.0)
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
                            nn.Linear(self.width, int(nn_out/3)))

    # Euler Angles Model accel model
    self.euler = nn.Sequential(nn.Linear(self.width, self.width),
                            copy.deepcopy(self.activation),
                            nn.Dropout(p=self.dropout),
                            nn.Linear(self.width, self.width),
                            copy.deepcopy(self.activation),
                            nn.Dropout(p=self.dropout),
                            nn.Linear(self.width, int(nn_out/3)))

    # Linear accel model
    self.linear = nn.Sequential(nn.Linear(self.width, self.width),
                            copy.deepcopy(self.activation),
                            nn.Dropout(p=self.dropout),
                            nn.Linear(self.width, self.width),
                            copy.deepcopy(self.activation),
                            nn.Dropout(p=self.dropout),
                            nn.Linear(self.width, int(nn_out/3)))

  def forward(self, x):

    x = self.main(x)
    angular = self.angular(x)
    euler = self.euler(x)
    linear = self.linear(x)

    if self.prob:
      means     = torch.cat((angular[:,:3], euler[:,:3], linear[:,:3]), 1)
      variances = torch.cat((angular[:,3:], euler[:,3:], linear[:,3:]), 1)
      x = torch.cat((means, variances), 1)
    else:
      x = torch.cat((angular, euler, linear), 1)

    return x
