import torch
import numpy as np
from .model import DynamicsModel
import hydra


class ResidualModel(DynamicsModel):
    def __init__(self, env, sub_model):
        """
        Residual model takes the base environment for the actions then uses a linear or deep model to learn the disturbance
        s_t+1 = env(s,a) + model(s,a)
        """
        super(ResidualModel, self).__init__()
        self.env = env
        self.model = hydra.utils.instantiate(sub_model)

    def forward(self, x):
        self.sub_model.forward(x)

    def reset(self):
        self.env.reset()
        self.model.reset()

    def preprocess(self, dataset):
        self.sub_model.preprocess(dataset)

    def postprocess(self, dX):
        self.sub_model.postprocess(dX)

    def train_cust(self, dataset, **params):
        self.sub_model.train_cust(dataset, **params)

    def predict(self, X, U):
        pred_model = self.sub_model.predict(X, U)
        self.env.set_state(X)
        pred_env = self.env.step(U)
        y = pred_env + pred_model
        return y
