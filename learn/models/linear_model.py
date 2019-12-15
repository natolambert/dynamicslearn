import torch
import numpy as np
from .model import DynamicsModel

class LinearModel(DynamicsModel):
    def __init__(self, cfg):
        """
        Model for a simple linear prediction of the change in state. Consider the following optimization problem:
            (s_t+1 - s_t) = As_t + Bu_t
        - hopefully this will be configurable with different linear solvers beyond least squares.
        - For instance, least squares is only looking to model the residual on the state error, but something like total
            least squares would assume there is noise at the measured state / action too
        - Or something like logistic regression which is less aggressive to outliers (which we definitely have)
        """
        super(LinearModel, self).__init__(cfg)

    def forward(self, x):
        raise NotImplementedError("Subclass must implement this function")

    def reset(self):
        raise NotImplementedError("Subclass must implement this function")

    def preprocess(self, dataset):
        raise NotImplementedError("Subclass must implement this function")

    def postprocess(self, dX):
        raise NotImplementedError("Subclass must implement this function")

    def train_cust(self, dataset, train_params, gradoff=False):
        raise NotImplementedError("Subclass must implement this function")

    def predict(self, X, U):
        raise NotImplementedError("Subclass must implement this function")

    def save_model(self, filepath):
        torch.save(self, filepath)
