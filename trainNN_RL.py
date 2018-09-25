# minimal file to load existing NN and continue training with new data

from dynamics import *
import pickle
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from model_general_nn import GeneralNN, predict_nn
import torch
from torch.nn import MSELoss
import time
import datetime
from model_split_nn import SplitModel
import os
# Plotting
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
import csv
from enum import Enum

# load network
# model_name = '_models/current_best/2018-08-23--14-21-35.9||w=150e=250lr=7e-06b=32d=2018_08_22_cf1_hover_p=True.pth'
# newNN = torch.load(model_name)
#
# with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
#     normX,normU,normdX = pickle.load(pickle_file)

# new_data = []
# # load new data
# with open('_logged_data_autonomous/Aug_24th/flight_log-20180824-153552.csv', "rb") as csvfile:
#     new_data = np.loadtxt(csvfile, delimiter=",")

# X_rl, U_rl, dX_rl = stack_dir("/rostime_false/")

delta = True
input_stack = 4
# X_rl, U_rl, dX_rl = stack_dir("/sep9_150_ng/", delta = delta, input_stack = input_stack)
# X_rl_2, U_rl_2, dX_rl_2 = stack_dir("/sep9_150_ng2/", delta = delta, input_stack = input_stack)
# X_rl_3, U_rl_3, dX_rl_3 = stack_dir("/sep10_150_ng/", delta = delta, input_stack = input_stack)
# X_rl_3, U_rl_3, dX_rl_3 = stack_dir("/sep10_150_ng2/", delta = delta, input_stack = input_stack)
# X_rl_4, U_rl_4, dX_rl_4 = stack_dir("/sep10_150_ng3/", delta = delta, input_stack = input_stack)

X_rl, U_rl, dX_rl = stack_dir("/sep12_float/", delta = delta, input_stack = input_stack)
X_rl_2, U_rl_2, dX_rl_2 = stack_dir("/sep13_150/", delta = delta, input_stack = input_stack)
# X_rl_3, U_rl_3, dX_rl_3 = stack_dir("/sep13_150_2/", delta = delta, input_stack = input_stack)
# X_rl_4, U_rl_4, dX_rl_4 = stack_dir("/sep14_150_2/", delta = delta, input_stack = input_stack, takeoff=True)
# X_rl_5, U_rl_5, dX_rl_5 = stack_dir("/sep14_150_3/", delta = delta, input_stack = input_stack, takeoff=True)
# X_rl_6, U_rl_6, dX_rl_6 = stack_dir("/sep14_150_4/", delta = delta, input_stack = input_stack, takeoff=True)


# quit()
X = X_rl
U = U_rl
dX = dX_rl
X = np.concatenate((X_rl,X_rl_2),axis=0)
U = np.concatenate((U_rl,U_rl_2),axis=0)
dX = np.concatenate((dX_rl,dX_rl_2),axis=0)

# X = np.concatenate((X_rl,X_rl_2,X_rl_3),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3),axis=0)

# X = np.concatenate((X_rl,X_rl_2,X_rl_3,X_rl_4),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3, U_rl_4),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3, dX_rl_4),axis=0)
#
# X = np.concatenate((X_rl,X_rl_2,X_rl_3,X_rl_4, X_rl_5),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3, U_rl_4, U_rl_5),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3, dX_rl_4, dX_rl_5),axis=0)

# X = np.concatenate((X_rl,X_rl_2,X_rl_3,X_rl_4, X_rl_5, X_rl_6),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3, U_rl_4, U_rl_5, U_rl_6),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3, dX_rl_4, dX_rl_5, dX_rl_6),axis=0)

# print(len(X))
#

# print(np.shape(X))
# print(np.shape(dX))
# X = np.delete(X,[0,1,2,10,11,12],1)
# dX = np.delete(dX, [0,1,2],1)
# dX = np.delete(dX, [2,5,8],1)
# print(np.shape(X))
# print(np.shape(dX))

# print(np.unique(U))
#
# X = X_rl_3
# dX = dX_rl_3
# U = U_rl_3
# pitch = X[:,3]
# roll = X[:,4]
# MSE = ((pitch)**2+(roll**2)).mean()
# print(MSE)
# quit()


w = 500     # Network width
e = 60 #450      # number of epochs
b  = 32

     # batch size was 32
lr = .0003 #.002  # .003    # learning rate was .005, .01 for adam
# lr = 3e-3    # learning rate
dims = [0,1,2,3,4,5,6,7,8]
depth = 3
prob_flag = True
#
dX = dX#[:, 3:6]
# print(np.mean(X[:,6:], axis=0))
# quit()
X = X #[:, 3:6]

# delta = True
# if delta:
#     data =
# else:
#     data =

# remove repeated euler angles
# X = X[np.all(dX[:,:] !=0, axis=1)]
# U = U[np.all(dX[:,:] !=0, axis=1)]
# dX = dX[np.all(dX[:,:] !=0, axis=1)]
# print(np.shape(dX))

n, d_o = np.shape(dX)
_, dx = np.shape(X)
_, du = np.shape(U)
print(du)
print(dx)
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        # m.bias.data.fill_(0.01)

# Initialize
newNN = GeneralNN(n_in_input = du,
                    n_in_state = len(dims)*input_stack,
                    hidden_w=w,
                    n_out = d_o, #len(dims),
                    state_idx_l=dims,
                    prob=prob_flag,
                    input_mode = 'Stacked Data',
                    pred_mode = 'Delta State',
                    depth=depth,
                    activation="Swish",
                    B = 1.0,
                    outIdx = dims,
                    dropout=0.0,
                    split_flag = False)

newNN.features.apply(init_weights)

# Checks the loss function scalers because these were wrong for a long time
if prob_flag:
    print('REMINDER: Did you check the scalars?')
    print(np.std(dX,axis=0))
    newNN.loss_fnc.scalers = torch.Tensor(np.std(dX,axis=0))
    # newNN.loss_fnc.scalers = torch.Tensor([.1, .1, .01, 10, 10,.5, 1, 1, 1])
    print(newNN.loss_fnc.scalers)
    print('Do the two values above match?')

# Train
acctest, acctrain = newNN.train_cust((X, U, dX),
                    learning_rate = lr,
                    epochs=e,
                    batch_size = b,
                    optim="Adam",
                    split=0.8)

min_err = min(acctrain)
min_err_test = min(acctest)
ax1 = plt.subplot(211)
ax1.set_yscale('log')
ax1.plot(acctest, label = 'Test Accurcay')
plt.title('Test  Train Accuracy')
ax2 = plt.subplot(212)
# ax2.set_yscale('log')
ax2.plot(acctrain, label = 'Train Accurcay')
plt.title('Training Accuracy')
ax1.legend()
plt.show()
# print(acc[::10])
# Saves NN params
dir_str = str('_models/temp_reinforced/')
date_str = str(datetime.datetime.now())[:-5]
data_name = '_CONF_'
date_str = date_str.replace(' ','--').replace(':', '-')
info_str = "--Min error"+ str(min_err_test)+ "--w=" + str(w) + "e=" + str(e) + "lr=" + str(lr) + "b=" + str(b) + "de=" + str(depth) + "d=" + str(data_name) + "p=" + str(prob_flag)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')
print('Saving model to', model_name)
normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"--normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)

quit()
delta = True
pred_dims = [0,1,2,3,4,5,6,7,8]
n = np.shape(xs)[0]
print(n)
# Now need to iterate through all data and plot
predictions_1 = np.empty((0,np.shape(xs)[1]))
print(np.shape(predictions_1))
for (dx, x, u) in zip(dxs, xs, us):
    # grab prediction value
    # pred = model.predict(x,u)
    pred = predict_nn(nn1,x,u, pred_dims)

    # print(np.shape(pred))
    #print('prediction: ', pred, ' x: ', x)
    if delta:
      pred = pred - x
    # print(pred)
    predictions_1 = np.append(predictions_1, pred.reshape(1,-1),  axis=0)
    # print(pred)
print(np.shape(predictions_1))

# Return evaluation along each dimension of the model
MSE = np.zeros(len(pred_dims))
print(np.shape(dxs))

for i, d in enumerate(pred_dims):
    se = (predictions_1[:,d] - dxs[:,i])**2
    # se = (predictions_1[:,d] - dxs[:,i])**2
    mse = np.mean(se)
    MSE[i] = mse
print('MSE Across Learned Dimensions')
print(MSE)


quit()

delta = True
sort = True

# newNN.eval()

model_1 = '_models/temp_reinforced/2018-09-07--07-33-00.2--Min error-89.02649255701013||w=500e=600lr=0.001b=20de=2d=_SEP6p=True.pth'
# Load a NN model with:
nn1 = torch.load(model_1)
nn1.training = False
nn1.eval()
with open(model_1[:-4]+'||normparams.pkl', 'rb') as pickle_file:
    normX1,normU1,normdX1 = pickle.load(pickle_file)

X_rl, U_rl, dX_rl = stack_dir("Sep_5_rand/")
# # X_rl_2, U_rl_2, dX_rl_2 = stack_dir("Aug_29th_125_2/")
#
#
# # xs = np.concatenate((X_rl,X_rl_2),axis=0)
# # us = np.concatenate((U_rl,U_rl_2),axis=0)
# # dxs = np.concatenate((dX_rl,dX_rl_2),axis=0)
xs = X_rl
us = U_rl
dxs = dX_rl
dims = [0,1,2] #,3,4,5,6,7,8]
# Now need to iterate through all data and plot
predictions_1 = np.empty((0,np.shape(xs)[1]))
model_dims = dims
for (dx, x, u) in zip(dxs, xs, us):
    # grab prediction value
    # pred = model.predict(x,u)
    pred = predict_nn(nn1,x,u, model_dims)
    # print(np.shape(pred))
    #print('prediction: ', pred, ' x: ', x)
    if delta:
      pred = pred - x
    predictions_1 = np.append(predictions_1, pred.reshape(1,-1),  axis=0)

# 0  1  2  3  4  5  6  7  8
# wx wy wz p  r  y  lx ly lz
dim = 2
# Grab correction dimension data
if delta:
    ground_dim = dxs[:, dim]
else:
    ground_dim = xs[:,dim]
pred_dim_1 = predictions_1[:, dim]


# Sort with respect to ground truth
if sort:
    data = zip(ground_dim,pred_dim_1)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort, pred_dim_sort_1 = zip(*data)


font = {'family' : 'normal',
    'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2.5)

ax1 = plt.subplot(111)
plt.title('Comparison of Deterministic and Probablistic One-Step Predictions')

x = np.size(ground_dim)
print(x)
time = np.linspace(0, int(x)-1, int(x))
ts = 4*10**-3
time = np.array([t*ts for t in time])

if sort:
    ax1.plot(ground_dim_sort, label='Ground Truth', color='k')
    ax1.plot(pred_dim_sort_1, label='Model 1 Prediction', linewidth=.8)#, linestyle=':')
    ax1.set_xlabel('Sorted Datapoints')
    ax1.set_ylabel('Roll Step (Degrees)')
else:
    ax1.plot(time-34, ground_dim, label='Ground Truth', color='k')
    ax1.plot(time-34, pred_dim_1, label='Probablistic Model Prediction') #, linestyle=':')
    # ax1.set_xlim([34,35])
    ax1.set_xlim([0,.5])
    ax1.set_ylim([-5,5])
    ax1.set_xticks(np.arange(0., .5, 0.05))
    ax1.set_yticks(np.arange(-5., 5., 1.))
    plt.grid(True, ls= 'dashed')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Roll Step (Degrees)')


ax1.legend()
plt.show()
