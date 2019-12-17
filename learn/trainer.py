import os
import sys

sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *
from learn.utils.nn import *

# neural nets
from learn.models.model_general_nn import GeneralNN
from learn.models.model_ensemble_nn import EnsembleNN
from learn.models.linear_model import LinearModel

# Torch Packages
import torch

# timing etc
import os
import hydra

# Plotting
import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)


def save_file(object, filename):
    path = os.path.join(os.getcwd(), filename)
    log.info(f"Saving File: {filename}")
    torch.save(object, path)


def create_model_params(df, model_cfg):
    # only take targets from robot.yaml
    target_keys = []
    for typ in model_cfg.delta_state_targets:
        target_keys.append(typ + '_0dx')
    for typ in model_cfg.true_state_targets:
        target_keys.append(typ + '_1fx')

    # grab variables
    history_states = df.filter(regex='tx')
    history_actions = df.filter(regex='tu')

    # add extra inputs like objective function
    extra_inputs = []
    if model_cfg.extra_inputs:
        for extra in model_cfg.extra_inputs:
            extra_inputs.append(extra)

    # trim past states to be what we want
    history = int(history_states.columns[-1][-3])
    if history > model_cfg.history:
        for i in range(history, model_cfg.history, -1):
            str_remove = str(i) + 't'
            for state in history_states.columns:
                if str_remove in state:
                    history_states.drop(columns=state, inplace=True)
            for action in history_actions.columns:
                if str_remove in action:
                    history_actions.drop(columns=action, inplace=True)

    # ignore states not helpful to prediction
    for ignore in model_cfg.ignore_in:
        for state in history_states.columns:
            if ignore in state:
                history_states.drop(columns=state, inplace=True)

    params = dict()
    params['targets'] = df.loc[:, target_keys]
    params['states'] = history_states
    params['inputs'] = history_actions
    # TODO add extra inputs to these parameters

    return params


def params_to_training(data):
    X = data['states'].values
    U = data['inputs'].values
    dX = data['targets'].values
    return X, U, dX


def train_model(X, U, dX, model_cfg):
    log.info("Training Model")
    train_log = dict()
    # nn_params = {  # all should be pretty self-explanatory
    #     'dx': dx,
    #     'du': du,
    #     'dt': dt,
    #     'hid_width': model_cfg.training.hid_width,
    #     'hid_depth': model_cfg.training.hid_depth,
    #     'bayesian_flag': model_cfg.training.probl,
    #     'activation': Swish(),  # TODO use hydra.utils.instantiate
    #     'dropout': model_cfg.training.extra.dropout,
    #     'split_flag': False,
    #     'ensemble': model_cfg.ensemble
    # }
    #s
    # train_params = {
    #     'epochs': model_cfg.optimizer.epochs,
    #     'batch_size': model_cfg.optimizer.batch,
    #     'optim': model_cfg.optimizer.name,
    #     'split': model_cfg.optimizer.split,
    #     'lr': model_cfg.optimizer.lr,  # bayesian .00175, mse:  .0001
    #     'lr_schedule': model_cfg.optimizer.lr_schedule,
    #     'test_loss_fnc': [],
    #     'preprocess': model_cfg.optimizer.preprocess,
    # }
    train_log['model_params'] = model_cfg.params
    model = hydra.utils.instantiate(model_cfg)
    acctest, acctrain = model.train_cust((X, U, dX), model_cfg.params)

    if model_cfg.params.training.ensemble:
        min_err = np.min(acctrain, 0)
        min_err_test = np.min(acctest, 0)
    else:
        min_err = np.min(acctrain)
        min_err_test = np.min(acctest)

    train_log['testerror'] = acctest
    train_log['trainerror'] = acctrain
    train_log['min_trainerror'] = min_err
    train_log['min_testerror'] = min_err_test

    return model, train_log


######################################################################
@hydra.main(config_path='conf/trainer.yaml')
def trainer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ######################################################################
    log.info('Training a new model')

    data_dir = cfg.load.base_dir

    avail_data = os.path.join(os.getcwd()[:os.getcwd().rfind('outputs') - 1] + f"/ex_data/SAS/{cfg.robot}.csv")
    if os.path.isfile(avail_data):
        df = pd.read_csv(avail_data)
        log.info(f"Loaded preprocessed data from {avail_data}")
    else:
        if cfg.robot == 'iono':
            df, log_load = preprocess_iono(data_dir, cfg.load)
        else:
            df, log_load = preprocess_cf(data_dir, cfg.load)
        msg = f"Loading Data"
        if 'dir' in log_load is not None:
            msg += f", dir={log_load['dir']}"
        if 'num_files' in log_load is not None:
            msg += f", num_files={log_load['num_files']}"
        if 'datapoints' in log_load:
            msg += f", datapoints={log_load['datapoints']}"
        log.info(msg)

    data = create_model_params(df, cfg.model)

    X, U, dX = params_to_training(data)

    model, train_log = train_model(X, U, dX, cfg.model)
    model.store_training_lists(list(data['states'].columns),
                               list(data['inputs'].columns),
                               list(data['targets'].columns))

    msg = "Trained Model..."
    msg += "Prediction List" + str(list(data['targets'].columns)) + "\n"
    msg += "Min test error: " + str(train_log['min_testerror']) + "\n"
    msg += "Mean Min test error: " + str(np.mean(train_log['min_testerror'])) + "\n"
    msg += "Min train error: " + str(train_log['min_trainerror']) + "\n"
    log.info(msg)

    if cfg.model.training.plot_loss:
        ax1 = plt.subplot(211)
        ax1.plot(train_log['testerror'], label='Test Loss')
        plt.title('Test Loss')
        ax2 = plt.subplot(212)
        ax2.plot(train_log['trainerror'], label='Train Loss')
        plt.title('Training Loss')
        ax1.legend()
        # plt.show()
        plt.savefig(os.path.join(os.getcwd() + '/modeltraining.pdf'))

    # Saves NN params
    if cfg.save:
        save_file(model, cfg.model.name + '.pth')

        normX, normU, normdX = model.getNormScalers()
        save_file((normX, normU, normdX), cfg.model.name + "_normparams.pkl")

        # Saves data file
        save_file(data, cfg.model.name + "_data.pkl")

    log.info(f"Saved to directory {os.getcwd()}")


if __name__ == '__main__':
    sys.exit(trainer())
