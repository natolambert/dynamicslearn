import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from _activation_swish import Swish

# module that sends the [0, nn_in - split] inputs through one network and the [split, nn_in] inputs through another
class SplitModel(nn.Module):

  def __init__(self, nn_in, nn_out, out_split, prob=True, depth=2, width=150, activation = 'swish', dropout = 0.02):
    super(SplitModel, self).__init__()

    self.nn_in      = nn_in
    self.nn_out     = nn_out
    self.out_split  = out_split
    self.prob       = prob
    self.depth      = depth
    self.width      = width
    self.dropout    = dropout

    if activation.lower() == 'swish':
      self.activation = Swish(B = 1.0)
    elif activation.lower() == 'relu':
      self.activation = nn.ReLU()

    angle_modules = []
    euler_modules = []

    # Input layers
    angle_modules.append(nn.Linear(self.nn_in, self.width))
    angle_modules.append(copy.deepcopy(self.activation))
    angle_modules.append(nn.Dropout(p=self.dropout))

    euler_modules.append(nn.Linear(self.nn_in, self.width))
    euler_modules.append(copy.deepcopy(self.activation))
    euler_modules.append(nn.Dropout(p=self.dropout))

    # hidden layers
    hid_layer = [nn.Linear(self.width, self.width), copy.deepcopy(self.activation), nn.Dropout(p=self.dropout)]
    for _ in range(0,self.depth-1):
      angle_modules.extend(hid_layer[:])
      euler_modules.extend(hid_layer[:])

    # Output layers
    angle_modules.append(nn.Linear(self.width, self.nn_out - self.out_split))
    euler_modules.append(nn.Linear(self.width, self.out_split))

    # Sequential Object
    self.angles_model = nn.Sequential(*angle_modules)
    self.eulers_model = nn.Sequential(*euler_modules)

  def forward(self, x):

    angle_out = self.angles_model(x)
    euler_out = self.eulers_model(x)

    if self.prob:
      means     = torch.cat((angle_out[:,:int((self.out_split)/2)], euler_out[:,:int((self.out_split)/2)]), 1)
      variances = torch.cat((angle_out[:,int((self.nn_out - self.out_split)/2):], euler_out[:,int((self.nn_out - self.out_split)/2):]), 1)
      x = torch.cat((means, variances), 1)
    else:
      x = torch.cat((angle_out, euler_out), 1)

    return x
