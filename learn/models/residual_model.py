import torch
import numpy as np
from .model import DynamicsModel


class ResidualModel(DynamicsModel):
    def __init__(self, cfg):
        """
        Residual model takes the base environment for the actions then uses a linear or deep model to learn the disturbance
        s_t+1 = env(s,a) + model(s,a)
        """
        super(ResidualModel, self).__init__(cfg)

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
