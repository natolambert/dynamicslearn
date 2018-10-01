# Import project files
import utils_data

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
from Swish import Swish
from model_split_nn import SplitModel
from model_split_nn_v2 import SplitModel2
import matplotlib.pyplot as plt
from lossfnc_pnngaussian import PNNLoss_Gaussian

'''
This file is in the works for an object to easily create an ensemble model. These
  models will be used heavily for offline bootstrapping of policies and controllers.

  Some features that should be implemented:
  - Easily predict the output of the network with forward()
  - change whether or not the models are trained on separate datasets sampled with
    replacement or relying on SGD / random initializations to get different networks
    (Do I have to check random seeds for this?)

'''
