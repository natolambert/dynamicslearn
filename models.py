# Compatibility Python 3

# Import project files

# Import External Packages
import numpy as np
import math
import torch
from sklearn.model_selection import train_test_split
from sklearn import linear_modle

class LearningModel:
    # class for dynamics learning model
    # TODO


class LeastSquares(LearningModel):
    # fits gathered data to the form
    # x_(t+1) = Ax + Bu
    def __init__(self):
        self.reg = linear_model.LinearRegression()

    def train(self, next_states, states_actions_prev):
        self.reg.fit()

    def predict(self, states_actions_prev):

class NeuralNet(LearningModel):
    # TODO

class GaussianProcess(LearningModel):
    # TODO

def ComputeAccuracy(truth,prediction):
