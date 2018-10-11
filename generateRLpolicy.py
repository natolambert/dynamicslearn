# Our infrastucture files
from utils_data import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN, predict_nn
from model_split_nn import SplitModel
from _activation_swish import Swish
from model_ensemble_nn import EnsembleNN

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib

import argparse

'''
This file will be the hub location for learning a control policy on top of a
   learned dynamcis model. As implementation proceeds, we will take in more
   arguments from the command line that will attempt to learn control policies
   a = pi(s). This type of policy has potential to be substantially less
   computationally intensive than MPC.
'''

######################################################################
def main():
    # adding arguments to make code easier to work with
    parser = argparse.ArgumentParser(description='Run function to train or run a RL policy on a dynamics model')
    parser.add_argument('policy', type=str,
                        choices = ['Value', 'Policy'],
                        help='choose which type of policy search to use.')
    parser.add_argument('--log', action='store_true',
                        help='a flag for storing a training log in a txt file')
    parser.add_argument('--noprint', action='store_false',
                        help='turn off printing in the terminal window for epochs')
    parser.add_argument('--plot', action='store_true',
                        help='plots information for easy analysis')

    args = parser.parse_args()

    log = args.log
    noprint = args.noprint
    ensemble = args.ensemble

######################################################################


if __name__ == "__main__":
    print('\n---------------------------------------------------')
    print('Running file generateRLpolicy')
    print('\n')
    main()
