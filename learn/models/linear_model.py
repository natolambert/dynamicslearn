import torch
import numpy as np
from .model import DynamicsModel
from ..utils.nn import ModelDataHandler
import hydra


class LinearModel(DynamicsModel):
    def __init__(self, **nn_params): #datahandler, solver=None, training=None, optimizer=None):
        """
        Model for a simple linear prediction of the change in state. Consider the following optimization problem:
            (s_t+1 - s_t) = As_t + Bu_t
        - hopefully this will be configurable with different linear solvers beyond least squares.
        - For instance, least squares is only looking to model the residual on the state error, but something like total
            least squares would assume there is noise at the measured state / action too
        - Or something like logistic regression which is less aggressive to outliers (which we definitely have)
        """
        super(LinearModel, self).__init__()
        self.data_handler = hydra.utils.instantiate(nn_params['datahandler'])  # ModelDataHandler(cfg)
        self.w = None
        self.solver = nn_params['training'].solver
        self.ensemble = False  # TODO try ensembling these

    def forward(self, x):
        if self.w is None:
            raise ValueError("Model Not Trained Yet, call model.train_cust(dataset, cfg)")
        x = np.matmul(x, self.w)
        return x

    def reset(self):
        print("Linear model does not need to reset")
        return

    def preprocess(self, dataset):
        inputs, outputs = self.data_handler.preprocess(dataset)
        return inputs, outputs

    def postprocess(self, dX):
        dX = self.data_handler.postprocess(dX)
        return dX

    def train_cust(self, dataset, train_params, ret_params=False):
        X = dataset[0]
        U = dataset[1]
        dX = dataset[2]
        # A = np.hstack((X, U))
        # b = dX
        inputs, outputs = self.preprocess(dataset)

        A = inputs
        b = outputs

        # Generate the weights of the least squares problem
        w, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        self.w = w
        res = np.mean((b-np.dot(A,w)**2))
        if ret_params:
            return w, (res, rank, s)
        else:
            return res, res

    def predict(self, X, U):
        normX, normU = self.data_handler.forward(X, U)
        val = np.concatenate((normX, normU), axis=1)
        y = self.forward(val)
        return self.data_handler.postprocess(y)
