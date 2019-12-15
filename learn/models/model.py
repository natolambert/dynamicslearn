import torch


class DynamicsModel:
    def __init__(self, cfg):
        """
        Parent class for different types of dynamics models
        """
        self.cfg = cfg

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
