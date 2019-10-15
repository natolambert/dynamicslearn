import os
import sys

sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *

# data packages

# neural nets

# Torch Packages
import torch

# timing etc
import os
import hydra

# Plotting

import logging

log = logging.getLogger(__name__)

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
