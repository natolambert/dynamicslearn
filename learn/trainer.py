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
    # if not model_cfg.params.true_state_targets == False:
    #     for typ in model_cfg.params.true_state_targets:
    #         target_keys.append(typ + '_1fx')

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
        mat = to_matrix(X.squeeze(), U, dX.squeeze(), model_cfg)
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
        X_t = X.squeeze()
        U_t = U
        dX_t = dX.squeeze()

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
    # quick_iono(df)
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

    # from learn.utils.plotly import plot_mses
    # plot_mses(save=True)



    data = create_model_params(df, cfg.model)

    X, U, dX = params_to_training(data)

    end_pts = np.where(df['term'].values == 1)
    init = np.add(np.where(df['term'].values == 1), 1).squeeze()
    all = np.concatenate(([0], init))

    model, train_log = train_model(X, U, dX, cfg.model)
    model.store_training_lists(list(data['states'].columns),
                               list(data['inputs'].columns),
                               list(data['targets'].columns))

    # mse = plot_test_train(model, (X, U, dX), variances=True)
    # torch.save((mse, cfg.model.params.training.cluster), 'cluster.dat')
    # log.info(f"MSE of test set predictions {mse}")

    msg = "Trained Model..."
    msg += "Prediction List" + str(list(data['targets'].columns)) + "\n"
    msg += "Min test error: " + str(train_log['min_testerror']) + "\n"
    msg += "Mean Min test error: " + str(np.mean(train_log['min_testerror'])) + "\n"
    msg += "Min train error: " + str(train_log['min_trainerror']) + "\n"
    log.info(msg)

    if False: #cfg.model.params.training.plot_loss:
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

    #compute MSEs on real data
    predictions = []
    true_trajs = []
    for i in range(len(all)-1):
        idx_start = all[i]
        idx_end = idx_start+30 #all[i+1]
        x0 = X[idx_start,:]
        uvec = U[idx_start:idx_end-1,:]
        true = X[idx_start+1:idx_end,:]
        pred = []
        xpred = x0
        for u in uvec:
            if cfg.model.params.training.ensemble:
                xpred = xpred + model.predict(xpred, u)[0]
            else:
                xpred = xpred + model.predict(xpred, u)
            pred.append(xpred)
        if len(true) < 29:
            continue
        else:
            predictions.append(pred)
            true_trajs.append(true)

    print(len(predictions))
    pred_vec = np.stack(predictions)
    true_vec = np.stack(true_trajs)
    errors = np.mean(np.square(true_vec-pred_vec),axis=-1)
    from learn.utils.plotly import plot_mses
    fig = plot_mses(errors=errors.tolist(),save=True)

    torch.save(errors, os.getcwd()+'/'+str(cfg.load.freq)+'.dat')
    # Saves NN params
    if cfg.save:
        save_file(model, cfg.model.params.name + '.pth')

        normX, normU, normdX = model.getNormScalers()
        save_file((normX, normU, normdX), cfg.model.params.name + "_normparams.pkl")

        # Saves data file
        save_file(data, cfg.model.params.name + "_data.pkl")

    log.info(f"Saved to directory {os.getcwd()}")

def plot_mse_err(mse_batch, save_loc=None, show=True, log_scale=True, title=None, y_min=.01, y_max=1e7,
                     legend=False):

    # assert setup, "Must run setup_plotting before this function"
    from utils.plotly import generate_errorbar_traces
    arrays = []
    keys = [k for k in mse_batch[0].keys()]
    for k in keys:
        temp = []
        for data in mse_batch:
            temp.append(data[k])
        arrays.append(np.stack(temp))


    colors_temp = ['rgb(12, 7, 134)', 'rgb(64, 3, 156)', 'rgb(106, 0, 167)',
                   'rgb(143, 13, 163)', 'rgb(176, 42, 143)', 'rgb(203, 71, 119)', 'rgb(224, 100, 97)',
                   'rgb(242, 132, 75)', 'rgb(252, 166, 53)', 'rgb(252, 206, 37)']
    traces_plot = []
    for n, (ar, k) in enumerate(zip(arrays, keys)):
        # temp
        # if n > 1:
        #     continue
        tr, xs, ys = generate_errorbar_traces(ar, xs=[np.arange(1, np.shape(ar)[1] + 1).tolist()], percentiles='66+90',
                                              color=color_dict_plotly[k],
                                              name=label_dict[k]+str(n))
        w_marker = []
        # for t in tr:
        m = add_marker(tr, color=color_dict_plotly[k], symbol=marker_dict_plotly[k], skip=30)
        # w_marker.append(m)
        [traces_plot.append(t) for t in m]

    layout = dict(  # title=title if title else f"Average Error over Run",
        xaxis={'title': 'Prediction Step'},  # 2e-9, 5
        yaxis={'title': 'Mean Squared Error', 'range': [np.log10(20e-6), np.log10(5)]},# 25]}, #
        # [np.log10(y_min), np.log10(y_max)]},
        yaxis_type="log",
        xaxis_showgrid=False, yaxis_showgrid=False,
        font=dict(family='Times New Roman', size=50, color='#000000'),
        height=800,
        width=1500,
        plot_bgcolor='white',
        showlegend=legend,
        margin=dict(r=0, l=0, b=10, t=1),

        legend={'x': .01, 'y': .98, 'bgcolor': 'rgba(50, 50, 50, .03)',
                'font': dict(family='Times New Roman', size=30, color='#000000')}
    )

    fig = {
        'data': traces_plot,
        # 'layout': layout
    }

    import plotly.io as pio
    fig = go.Figure(fig)
    fig.update_layout(layout)
    if show: fig.show()
    fig.write_image(save_loc + ".pdf")

    return fig

if __name__ == '__main__':
    sys.exit(trainer())
