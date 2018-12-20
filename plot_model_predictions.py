__author__ = 'Nathan Lambert'
__version__ = '1.0'

# Our infrastucture files
from utils_data import *
from utils_nn import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN
from model_split_nn import SplitModel

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

print('\n')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
print('Running... plot_predictions.py' + date_str +'\n')


load_params = {
    'delta_state': True,                # normally leave as True, prediction mode
    # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'include_tplus1': True,
    # trims high vbat because these points the quad is not moving
    'trim_high_vbat': 4050,
    # If not trimming data with fast log, need another way to get rid of repeated 0s
    'takeoff_points': 180,
    # if all the euler angles (floats) don't change, it is not realistic data
    'trim_0_dX': True,
    'find_move': True,
    # if the states change by a large amount, not realistic
    'trime_large_dX': True,
    # Anything out of here is erroneous anyways. Can be used to focus training
    'bound_inputs': [20000, 65500],
    # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'stack_states': 3,
    # looks for sharp changes to tthrow out items post collision
    'collision_flag': False,
    # shuffle pre training, makes it hard to plot trajectories
    'shuffle_here': False,
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery': True,                   # if battery voltage is in the state data
    # adds a column to the dataframe tracking end of trajectories
    'terminals': True,
    'fastLog': True,                   # if using the software with the new fast log
    # Number of times the control freq you will be using is faster than that at data logging
    'contFreq': 1
}


dir_list = ["_newquad1/publ_data/c50_samp300_rand/",
            "_newquad1/publ_data/c50_samp300_roll1/",
            "_newquad1/publ_data/c50_samp300_roll2/",
            "_newquad1/publ_data/c50_samp300_roll3/",
            "_newquad1/publ_data/c50_samp300_roll4/"]
dir_list = ["_newquad1/publ_data/c25_samp300_rand/",
            "_newquad1/publ_data/c25_samp300_roll1/",
            "_newquad1/publ_data/c25_samp300_roll2/",
            "_newquad1/publ_data/c25_samp300_roll3/",
            "_newquad1/publ_data/c25_samp300_roll4/"]

# dir_list = ["_newquad1/publ_data/c25_samp300_roll4/"]

df = load_dirs(dir_list, load_params)

data_params = {
    # Note the order of these matters. that is the order your array will be in
    'states': ['omega_x0', 'omega_y0', 'omega_z0',
               'pitch0',   'roll0',    'yaw0',
               'lina_x0',  'lina_y0',  'lina_z0',
               'omega_x1', 'omega_y1', 'omega_z1',
               'pitch1',   'roll1',    'yaw1',
               'lina_x1',  'lina_y1',  'lina_z1',
               'omega_x2', 'omega_y2', 'omega_z2',
               'pitch2',   'roll2',    'yaw2',
               'lina_x2',  'lina_y2',  'lina_z2'],
    # 'omega_x3', 'omega_y3', 'omega_z3',
    # 'pitch3',   'roll3',    'yaw3',
    # 'lina_x3',  'lina_y3',  'lina_z3'],

    'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
               'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
               'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],  # 'vbat'],
    # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

    'targets': ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                'd_pitch', 'd_roll', 'd_yaw',
                't1_lina_x', 't1_lina_y', 't1_lina_z'],

    'battery': True                    # Need to include battery here too
}
data_params = {
    # Note the order of these matters. that is the order your array will be in
    'states': ['omega_x0', 'omega_y0', 'omega_z0',
               'pitch0',   'roll0',    'yaw0',
               'lina_x0',  'lina_y0',  'lina_z0',
               'omega_x1', 'omega_y1', 'omega_z1',
               'pitch1',   'roll1',    'yaw1',
               'lina_x1',  'lina_y1',  'lina_z1',
               'omega_x2', 'omega_y2', 'omega_z2',
               'pitch2',   'roll2',    'yaw2',
               'lina_x2',  'lina_y2',  'lina_z2'],
    # 'omega_x3', 'omega_y3', 'omega_z3',
    # 'pitch3',   'roll3',    'yaw3',
    # 'lina_x3',  'lina_y3',  'lina_z3'],

    'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
               'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
               'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],  # 'vbat'],
    # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

    'targets': ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                'd_pitch', 'd_roll', 'd_yaw',
                't1_lina_x', 't1_lina_y', 't1_lina_z'],

    'battery': True                    # Need to include battery here too
}
X, U, dX = df_to_training(df, data_params)



# model_r0 = '_models/temp/2018-10-29--13-33-23.0--Min error-19.431143d=_50Hz_roll1_stack3_.pth'

# model_r1 = '_models/temp/2018-10-29--14-28-23.9--Min error-29.397131d=_50Hz_roll1_stack3_.pth'

# model_r2 = '_models/temp/2018-10-29--15-10-10.3--Min error-25.488068d=_50Hz_roll2_stack3_.pth'

# model_r3 = '_models/temp/2018-10-29--15-53-36.5--Min error-27.653141d=_50Hz_roll3_stack3_.pth'

# model_r4 = '_models/temp/2018-10-30--10-12-27.2--Min error-27.827824d=_50Hz_roll3_stack3_.pth'

# model_r4_tuned = '_models/temp/2018-10-31--06-56-01.0--Min error-28.411377d=_50Hz_roll3_stack3_.pth'
# model_r4_tuned2 = '_models/temp/2018-10-31--06-58-45.2--Min error-27.99523d=_50Hz_roll3_stack3_.pth'
# model_temp = '_models/temp/2018-11-08--20-32-42.6--Min error-41.806355d=_100Hz_roll3_stack3_.pth'

model_pll = '_models/temp/2018-12-14--10-47-41.7_plot_pll_stack3_.pth'
model_mse = '_models/temp/2018-12-14--10-51-10.9_plot_mse_stack3_.pth'
model_pll_ens = '_models/temp/2018-12-14--10-53-42.9_plot_pll_ensemble_stack3_.pth'
model_pll_ens_10 = '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth'
model_mse_ens = '_models/temp/2018-12-14--10-52-40.4_plot_mse_ensemble_stack3_.pth'

# model = model_temp
# # Load a NN model with:
# nn1 = torch.load(model)
# nn1.training = False
# nn1.eval()
# with open(model[:-4]+'--normparams.pkl', 'rb') as pickle_file:
#     normX1,normU1,normdX1 = pickle.load(pickle_file)


if False:
    train_params = {
        'epochs' : 1,
        'batch_size' : 32,
        'optim' : 'Adam',
        'split' : 0.999,
        'lr': .002,
        'lr_schedule' : [30,.6],
        'test_loss_fnc' : [],
        'preprocess' : True,
        'noprint' : False
    }
    _, trainloss = nn1.train_cust((X,U,dX), train_params, gradoff = True)
    print("Training Set Likelihood is: ", trainloss)
# quit()
xs = X
us = U
dxs = dX #[:,3:6]

# Note on dimensions
# 0  1  2  3  4  5  6  7  8
# wx wy wz p  r  y  lx ly lz

# Pred dims is the DIMENSIONS OF THE FULL State
delta = True
# print(np.amax((dxs),axis=0))
pred_dims = [0,1,2,3,4,5,6,7,8]
n, dimx = np.shape(xs)

def gather_predictions(model_dir, dataset, delta = True):
    """
    Takes in a dataset and returns a matrix of predictions for plotting.
    - model_dir of the form '_models/temp/... .pth'
    - dataset of the form (X, U, dX)
    - delta makes the plot of the change in state or global predictions 
    - note that predict_nn_v2 returns the global values, always
    """

    nn = torch.load(model_dir)
    nn.training = False
    nn.eval()
    with open(model_dir[:-4]+'--normparams.pkl', 'rb') as pickle_file:
        normX1, normU1, normdX1 = pickle.load(pickle_file)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    predictions_1 = np.empty((0, 9))#np.shape(X)[1]))
    for (dx, x, u) in zip(dX, X, U):

        # grab prediction value
        pred = predict_nn_v2(nn, x, u)

        #print('prediction: ', pred, ' x: ', x)
        # print(x.shape)
        # print(pred.shape)
        if delta:
            pred = pred - x[:9]
        # print(pred)
        predictions_1 = np.append(predictions_1, pred.reshape(1, -1),  axis=0)

    return predictions_1

def plot_euler_preds(model, dataset):
    """
    returns a 3x1 plot of the Euler angle predictions for a given model and dataset
    """

    predictions_1 = gather_predictions(model, dataset)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    dim = 3

    shift = 0
    # lx = int(n*.99)
    # Grab correction dimension data # for training :int(.8*n)

    if delta:
        ground_dim_1 = dX[:, 3]
        ground_dim_2 = dX[:, 4]
        ground_dim_3 = dX[:, 5]

    pred_dim_1 = predictions_1[:, 3]  # 3]
    pred_dim_2 = predictions_1[:, 4]  # 4]
    pred_dim_3 = predictions_1[:, 5]  # 5]
    global_dim_1 = X[:, 0+shift+dim]  # 3
    global_dim_2 = X[:, 1+shift+dim]  # 4
    global_dim_3 = X[:, 2+shift+dim]  # 5


    # Sort with respect to ground truth
    # data = zip(ground_dim,pred_dim_1, ground_dim_2, ground_dim_3)
    # data = sorted(data, key=lambda tup: tup[0])
    # ground_dim_sort, pred_dim_sort_1, ground_dim_sort_2, ground_dim_sort_3 = zip(*data)

    # sorts all three dimenions for YPR
    data = zip(ground_dim_1, pred_dim_1, global_dim_1)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_1, pred_dim_sort_1, global_dim_sort_1 = zip(*data)

    data = zip(ground_dim_2, pred_dim_2, global_dim_2)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_2, pred_dim_sort_2, global_dim_sort_2 = zip(*data)

    data = zip(ground_dim_3, pred_dim_3, global_dim_3)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_3, pred_dim_sort_3, global_dim_sort_3 = zip(*data)


    font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # plt.tight_layout()

    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)


    my_dpi = 300
    plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
    ax1.axhline(0, linestyle=':', color='r', linewidth=1)
    ax1.plot(ground_dim_sort_1, label='Ground Truth', color='k', linewidth=1.8)
    ax1.plot(pred_dim_sort_1, ':', label='Model Prediction',
            markersize=.9, linewidth=.8)  # , linestyle=':')
    # ax1.set_xlabel('Sorted Datapoints')
    ax1.set_ylabel('Pitch Step (Deg.)')
    # ax1.set_ylim([-5,5])
    # ax1.set_yticks(np.arange(-5,5.01,2.5))

    # ax1.legend()
    # plt.show()

    # plt.title('One Step Dim+1')
    ax2.axhline(0, linestyle=':', color='r', linewidth=1)
    ax2.plot(ground_dim_sort_2, label='Ground Truth', color='k', linewidth=1.8)
    ax2.plot(pred_dim_sort_2, ':', label='Model Prediction',
            markersize=.9, linewidth=.8)  # , linestyle=':')

    # ax2.set_xlabel('Sorted Datapoints')
    ax2.set_ylabel('Roll Step (Deg.)')
    # ax2.set_ylim([-5,5])
    # ax2.set_yticks(np.arange(-5,5.01,2.5))
    # ax2.set_yticklabels(["-5", "-2.5", "0", "2.5", "5"])

    # ax2.legend()
    # plt.show()

    # plt.title('One Step Dim+2')
    ax3.axhline(0, linestyle=':', color='r', linewidth=1)
    ax3.plot(ground_dim_sort_3, label='Ground Truth', color='k', linewidth=1.8)
    ax3.plot(pred_dim_sort_3, ':', label='Model Prediction',
            markersize=.9, linewidth=.8)  # , linestyle=':')

    ax3.set_xlabel('Sorted Datapoints')
    ax3.set_ylabel('Yaw Step (Deg.)')
    ax3.set_ylim([-5, 5])
    ax3.set_yticks(np.arange(-5, 5.01, 2.5))
    leg3 = ax3.legend(loc=8, ncol=2)
    for line in leg3.get_lines():
        line.set_linewidth(2.5)
    plt.show()



# Now need to iterate through all data and plot
# predictions_1 = np.empty((0,np.shape(xs)[1]))
# print(np.shape(predictions_1))
# for (dx, x, u) in zip(dxs, xs, us):

#     # grab prediction value
#     # pred = model.predict(x,u)
#     pred = predict_nn(nn1,x,u, pred_dims)
#     # print(np.shape(pred))
#     #print('prediction: ', pred, ' x: ', x)
#     if delta:
#       pred = pred - x
#     # print(pred)
#     predictions_1 = np.append(predictions_1, pred.reshape(1,-1),  axis=0)
#     # print(pred)
# print(np.shape(predictions_1))

# Return evaluation along each dimension of the model
# MSE = np.zeros(len(pred_dims))
# for i, d in enumerate([0,1,2,3,4,5,6,7,8]):
#     se = (predictions_1[:,d] - dxs[:,d])**2
#     # se = (predictions_1[:,d] - dxs[:,i])**2
#     mse = np.mean(se)
#     MSE[i] = mse
# print('MSE Across Learned Dimensions')
# print(MSE)

# predictions_pll = gather_predictions(model_pll, (X,U,dX))
# predictions_mse = gather_predictions(model_mse, (X,U,dX))
# predictions_pll_ens = gather_predictions(model_pll_ens, (X, U, dX))
predictions_pll_ens = gather_predictions(model_pll_ens_10, (X, U, dX))
# predictions_mse_ens = gather_predictions(model_mse_ens, (X, U, dX))

# Gather model parameters
print("Gathering MSEs and Likelihoods")
train_params = {
    'epochs': 1,
    'batch_size': 32,
    'optim': 'Adam',
    'split': 0.999,
    'lr': .002,
    'lr_schedule': [30, .6],
    'test_loss_fnc': 'pll',
    'preprocess': True,
    'noprint': False
}
# _, trainloss = torch.load(model_pll).train_cust((X, U, dX), train_params, gradoff=True)
# print("Training Set Likelihood is: ", trainloss)
# _, trainloss = torch.load(model_pll_ens_10).train_cust((X, U, dX), train_params, gradoff=True)
# print("Training Set Likelihood is: ", trainloss)

def print_error(predictions, dataset, mode = 'MSE'):
    """
    Takes in some predictions and an dataset and returns a test value of the predictions validity
    - modes are 'MSE' or 'Likelihood'
    """
    dX = dataset[2]
    dim = 4
    error = np.mean((predictions[:,4] - dX[:,4])**2,axis=0)
    print("MSE dim 4:", error)


# print_error(predictions_pll, (X,U,dX))
# print_error(predictions_mse, (X,U,dX))
# print_error(predictions_pll_ens, (X,U,dX))
# print_error(predictions_mse_ens, (X,U,dX))

dim = 4



# Gather test train splitted data
lx = int(np.shape(dX)[0]*.8)
data_train = zip(dX[:lx, dim], predictions_pll_ens[:lx, dim])
data_train = sorted(data_train, key=lambda tup: tup[0])
gt_sort_train, pred_sort_pll_train = zip(*data_train)

data_test = zip(dX[lx:, dim], predictions_pll_ens[lx:, dim])
data_test = sorted(data_test, key=lambda tup: tup[0])
gt_sort_test, pred_sort_pll_test = zip(*data_test)

# New plot
font = {'size': 22}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=4.5)

plt.tight_layout()

# plot for test train compare

with sns.axes_style("whitegrid"):
    fig = plt.figure()
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.5
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    plt.subplots_adjust(left=.1, right=1-.07,hspace = .28)

for ax in [ax1, ax2]:
    if ax == ax1:
        loc = matplotlib.ticker.MultipleLocator(base=int(lx/10))
    else:
        loc = matplotlib.ticker.MultipleLocator(base=int((np.shape(dX)[0]-lx)/10))
    ax.xaxis.set_major_locator(loc)

    ax.grid(b=True, which='major', color='k',
            linestyle='-', linewidth=1.2, alpha=.75)
    ax.grid(b=True, which='minor', color='b',
            linestyle='--', linewidth=.9, alpha=.5)
    ax.set_ylim([-6.0, 6.0])

fig.text(.02, .7, 'One Step Prediction, Pitch (deg)', rotation=90)
fig.text(.42, .04, 'Sorted Datapoints')

ax1.plot(gt_sort_train, label='Ground Truth', color='k', linewidth=1.8)
ax1.plot(pred_sort_pll_train, '-', label='Bayesian Model Training Data Prediction',
         markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
ax1.set_title("Training Data Predictions")

ax2.plot(gt_sort_test, label='Ground Truth', color='k', linewidth=1.8)
ax2.plot(pred_sort_pll_test, '-', label='Bayesian Model Validation DataPrediction',
         markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
ax2.set_title("Test Data Predictions")

plt.show()
# plot to compare 4 models

# Gather sorted predictions for all
# data = zip(dX[:, dim], predictions_pll[:, dim], predictions_mse[:, dim],
#            predictions_pll_ens[:, dim], predictions_mse_ens[:, dim])
# data = sorted(data, key=lambda tup: tup[0])
# gt_sort, pred_sort_pll, pred_sort_mse, pred_sort_pll_ens, pred_sort_mse_ens = zip(
#     *data)

# with sns.axes_style("whitegrid"):
#     fig =plt.figure()
#     plt.rcParams["axes.edgecolor"] = "0.15"
#     plt.rcParams["axes.linewidth"] = 1.5
#     ax1 = plt.subplot(221)
#     ax2 = plt.subplot(222)
#     ax3 = plt.subplot(223)
#     ax4 = plt.subplot(224)
#     plt.subplots_adjust(wspace=.18, left=.07, right=1-.07, hspace=.2)

# my_dpi = 300
# plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)

# for ax in [ax1,ax2,ax3,ax4]:
#     ax.grid(b=True, which='major', color='k',
#              linestyle='-', linewidth=1.2, alpha=.75)
#     ax.grid(b=True, which='minor', color='b',
#              linestyle='--', linewidth=.9, alpha=.5)
#     ax.set_ylim([-5,5])

# fig.text(.02, .7, 'One Step Prediction, Pitch (deg)', rotation = 90)
# fig.text(.42,.04, 'Sorted Datapoints')

# ax1.plot(gt_sort, label='Ground Truth', color='k', linewidth=1.8)
# ax1.plot(pred_sort_pll, ':', label='Bayesian Model Prediction',
#          markersize=.9, linewidth=.8)  # , linestyle=':')
# ax1.set_title("Bayesian Predictions")

# ax2.plot(gt_sort, label='Ground Truth', color='k', linewidth=1.8)
# ax2.plot(pred_sort_mse, ':', label='MSE Model Prediction',
#          markersize=.9, linewidth=.8)  # , linestyle=':')
# ax2.set_title("MSE Predictions")


# ax3.plot(gt_sort, label='Ground Truth', color='k', linewidth=1.8)
# ax3.plot(pred_sort_pll_ens, ':', label='Bayesian Ensemble Model Prediction',
#          markersize=.9, linewidth=.8)  # , linestyle=':')
# ax3.set_title("Bayesian Ensemble Predictions")

# ax4.plot(gt_sort, label='Ground Truth', color='k', linewidth=1.8)
# ax4.plot(pred_sort_mse_ens, ':', label='MSE Ensemble Model Prediction',
#          markersize=.9, linewidth=.8)  # , linestyle=':')
# ax4.set_title("MSE Ensemble Predictions")

# plt.show()

# TRY PLOTTING THE ERROR OF PREDCTIONS! if predictions aren't clear, maybe the change in predictions will be
# should also check if the MSE and PLL are both better at what they are trained at.... as expect, could mean we need to use the variance estimate if its worth doing
