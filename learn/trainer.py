import os
import sys

sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *
from learn.utils.nn import *
import learn.utils.matplotlib as u_p
from learn.utils.plotly import plot_test_train, plot_dist, quick_iono
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
import time


import logging

log = logging.getLogger(__name__)


def save_file(object, filename):
    path = os.path.join(os.getcwd(), filename)
    log.info(f"Saving File: {filename}")
    torch.save(object, path)


def create_model_params(df, model_cfg):
    # only take targets from robot.yaml
    target_keys = []
    if not model_cfg.params.delta_state_targets == False:
        for typ in model_cfg.params.delta_state_targets:
            target_keys.append(typ + '_0dx')
    if not model_cfg.params.true_state_targets == False:
        for typ in model_cfg.params.true_state_targets:
            target_keys.append(typ + '_1fx')

    # grab variables
    history_states = df.filter(regex='tx')
    history_actions = df.filter(regex='tu')

    # trim past states to be what we want
    history = int(history_states.columns[-1][-3])
    if history > model_cfg.params.history:
        for i in range(history, model_cfg.params.history, -1):
            str_remove = str(i) + 't'
            for state in history_states.columns:
                if str_remove in state:
                    history_states.drop(columns=state, inplace=True)
            for action in history_actions.columns:
                if str_remove in action:
                    history_actions.drop(columns=action, inplace=True)

    # add extra inputs like objective function
    extra_inputs = []
    if model_cfg.params.extra_inputs:
        for extra in model_cfg.params.extra_inputs:
            df_e = df.filter(regex=extra)
            extra_inputs.append(df_e)
            history_actions[extra] = df_e.values

    # ignore states not helpful to prediction
    for ignore in model_cfg.params.ignore_in:
        for state in history_states.columns:
            if ignore in state:
                history_states.drop(columns=state, inplace=True)

    params = dict()
    params['targets'] = df.loc[:, target_keys]
    params['states'] = history_states
    params['inputs'] = history_actions

    return params


def params_to_training(data):
    X = data['states'].values
    U = data['inputs'].values
    dX = data['targets'].values
    return X, U, dX


def train_model(X, U, dX, model_cfg, logged=False):
    if logged: log.info(f"Training Model on {np.shape(X)[0]} pts")
    start = time.time()
    train_log = dict()

    train_log['model_params'] = model_cfg.params
    model = hydra.utils.instantiate(model_cfg)

    if model_cfg.params.training.cluster > 0:
        h = model_cfg.params.history
        mat = to_matrix(X, U, dX, model_cfg)
        num_pts = np.shape(mat)[0]
        if num_pts < model_cfg.params.training.cluster:
            if logged: log.info(f"Not enough points to cluster to {model_cfg.params.training.cluster} yet.")
            X_t = X
            U_t = U
            dX_t = dX
        else:
            mat_r = cluster(mat, model_cfg.params.training.cluster)
            X_t, U_t, dX_t = to_Dataset(mat_r, dims=[model_cfg.params.dx * (h + 1), model_cfg.params.du * (h + 1),
                                                 model_cfg.params.dt])
    else:
        X_t = X
        U_t = U
        dX_t = dX

    acctest, acctrain = model.train_cust((X_t, U_t, dX_t), model_cfg.params)

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

    end = time.time()
    if logged: log.info(f"Trained Model in {end-start} s")
    return model, train_log


######################################################################
@hydra.main(config_path='conf/trainer.yaml')
def trainer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ######################################################################
    log.info('Training a new model')

    data_dir = cfg.load.fname  # base_dir

    avail_data = os.path.join(os.getcwd()[:os.getcwd().rfind('outputs') - 1] + f"/ex_data/SAS/{cfg.robot}.csv")
    if False: #os.path.isfile(avail_data):
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

    from scipy import stats
    quick_iono(df)
    # remove data 4 standard deviations away
    # df = df[(np.nan_to_num(np.abs(stats.zscore(df))) < 4).all(axis=1)]
    # plot_dist(df, x='roll_0tx', y='pitch_0tx', z='yaw_0tx')
    # data = create_model_params(df, cfg.model)
    # X = data['states'].values
    # U = data['inputs'].values
    # dX = data['targets'].values

    # x = torch.Tensor(np.hstack((X,U,dX))).numpy()
    # import faiss
    # # x = vectorized
    # niter = 50
    # ncentroids = 500
    # verbose = True
    # d = x.shape[1]
    # kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    # kmeans.train(x)
    #
    # # for i, v in enumerate(kmeans.centroids):
    # #     print(i)
    #
    # index = faiss.IndexFlatL2(d)
    # index.add(x)
    # D, I = index.search(kmeans.centroids, 1)
    # x_reduced = x[I, :].squeeze()
    # df.iloc[I.squeeze()]
    # plot_dist(df, x='roll_0tx', y='pitch_0tx', z='yaw_0tx')
    #
    # quit()




    data = create_model_params(df, cfg.model)

    X, U, dX = params_to_training(data)

    model, train_log = train_model(X, U, dX, cfg.model)
    model.store_training_lists(list(data['states'].columns),
                               list(data['inputs'].columns),
                               list(data['targets'].columns))

    mse = plot_test_train(model, (X, U, dX), variances=True)
    torch.save((mse, cfg.model.params.training.cluster), 'cluster.dat')

    log.info(f"MSE of test set predictions {mse}")
    msg = "Trained Model..."
    msg += "Prediction List" + str(list(data['targets'].columns)) + "\n"
    msg += "Min test error: " + str(train_log['min_testerror']) + "\n"
    msg += "Mean Min test error: " + str(np.mean(train_log['min_testerror'])) + "\n"
    msg += "Min train error: " + str(train_log['min_trainerror']) + "\n"
    log.info(msg)

    if cfg.model.params.training.plot_loss:
        plt.figure(2)
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
        save_file(model, cfg.model.params.name + '.pth')

        normX, normU, normdX = model.getNormScalers()
        save_file((normX, normU, normdX), cfg.model.params.name + "_normparams.pkl")

        # Saves data file
        save_file(data, cfg.model.params.name + "_data.pkl")

    log.info(f"Saved to directory {os.getcwd()}")


if __name__ == '__main__':
    sys.exit(trainer())
