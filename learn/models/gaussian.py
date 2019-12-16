import torch
import numpy as np
from .model import DynamicsModel


class GaussianProcess(DynamicsModel):
    def __init__(self, cfg):
        """
        Models the one state predictions with a Gaussian Process. If anyone can help implement this, let me know!
        """
        super(GaussianProcess, self).__init__(cfg)

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

