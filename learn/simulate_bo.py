import os
import sys

sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *
from learn.utils.nn import *

# data packages

# neural nets
from learn.model_general_nn import GeneralNN
from learn.model_ensemble_nn import EnsembleNN

# Torch Packages
import torch

# timing etc
import os
import hydra

# Plotting
import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)

import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression


def save_file(object, filename):
    path = os.path.join(os.getcwd(), filename)
    log.info(f"Saving File: {filename}")
    torch.save(object, path)


def model_rollout_variable(s0, model, controller):
    """

    :param s0:
    :param model:
    :param controller:
    :return:
    """


def model_rollout_fixed(s0, model, controller, k):
    """

    :param s0:
    :param model:
    :param controller:
    :param k:
    :return:
    """


def predict_with_var(model, state, action):
    """

    :param model:
    :param state:
    :param action:
    :return:
    """


######################################################################
@hydra.main(config_path='conf/simulate.yaml')
def simulate(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ######################################################################
    log.info('Running offline simulations on a model')

    data_dir = cfg.load.base_dir

    if cfg.model_dir is not None:
        model_path = cfg.model_dir
        log.info("")
    else:
        model_path = os.path.join(
            os.getcwd()[:os.getcwd().rfind('outputs') - 1] + f"/ex_data/models/{cfg.robot}.dat")
    log.info(f"Load model from {cfg.model_path}")
    model = torch.load(model_path)


    # Saves NN params
    if cfg.save:
        save_file(model, cfg.model.name + '.pth')

        normX, normU, normdX = model.getNormScalers()
        save_file((normX, normU, normdX), cfg.model.name + "_normparams.pkl")

        # Saves data file
        save_file(data, cfg.model.name + "_data.pkl")


if __name__ == '__main__':
    sys.exit(simulate())
