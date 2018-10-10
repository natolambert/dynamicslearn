__author__ = 'Nathan Lambert'
__version__ = '1.0'

# Our infrastucture files
from utils_data import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN, predict_nn
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

load_params ={
    'delta_state': True,
    'takeoff_points': 180,
    'trim_0_dX': True,
    'trime_large_dX': True,
    'bound_inputs': [20000,65500],
    'stack_states': 4,
    'collision_flag': False,
    'shuffle_here': False,
    'timestep_flags': [],
    'battery' : True
}

dir_list = ["_newquad1/150Hz_rand/"]
other_dirs = ["150Hz/sep13_150_2/","/150Hzsep14_150_2/","150Hz/sep14_150_3/"]
df = load_dirs(dir_list, load_params)

data_params = {
    'states' : [],
    'inputs' : [],
    'change_states' : [],
    'battery' : True
}

X, U, dX = df_to_training(df, data_params)

model_single = '_models/temp/2018-10-04--13-07-31.6--Min error-784.8953125d=_150Hz_newnet_.pth'

model_ensemble = '_models/temp/2018-10-04--13-06-23.9--Min error-767.918203125d=_150Hz_newnet_.pth'

model = model_single
# Load a NN model with:
nn1 = torch.load(model)
nn1.training = False
nn1.eval()
with open(model[:-4]+'--normparams.pkl', 'rb') as pickle_file:
    normX1,normU1,normdX1 = pickle.load(pickle_file)


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
for i, d in enumerate([3,4,5]):
    se = (predictions_1[:,d] - dxs[:,d])**2
    # se = (predictions_1[:,d] - dxs[:,i])**2
    mse = np.mean(se)
    MSE[i] = mse
print('MSE Across Learned Dimensions')
print(MSE)

# 0  1  2  3  4  5  6  7  8
# wx wy wz p  r  y  lx ly lz
dim = 3
shift = 0
lx = int(n*.8)
# Grab correction dimension data # for training :int(.8*n)
if True:
    if delta:
        ground_dim_1 = dxs[:lx, 0+dim]
        ground_dim_2 = dxs[:lx, 1+dim]
        ground_dim_3 = dxs[:lx, 2+dim]
    else:
        ground_dim_1 = xs[:, 0+dim]
        ground_dim_2 = xs[:, 1+dim]
        ground_dim_3 = xs[:, 2+dim]
    pred_dim_1 = predictions_1[:lx, 0+shift+dim] #3]
    pred_dim_2 = predictions_1[:lx, 1+shift+dim] #4]
    pred_dim_3 = predictions_1[:lx, 2+shift+dim] #5]
    global_dim_1 = xs[:lx,0+shift+dim] #3
    global_dim_2 = xs[:lx,1+shift+dim] # 4
    global_dim_3 = xs[:lx,2+shift+dim] # 5
else:
    if delta:
        ground_dim_1 = dxs[lx:, 0+dim]
        ground_dim_2 = dxs[lx:, 1+dim]
        ground_dim_3 = dxs[lx:, 2+dim]
    else:
        ground_dim_1 = xs[:, 0+dim]
        ground_dim_2 = xs[:, 1+dim]
        ground_dim_3 = xs[:, 2+dim]
    pred_dim_1 = predictions_1[lx:, 0+shift+dim] #3]
    pred_dim_2 = predictions_1[lx:, 1+shift+dim] #4]
    pred_dim_3 = predictions_1[lx:, 2+shift+dim] #5]
    global_dim_1 = xs[lx:,0+shift+dim] #3
    global_dim_2 = xs[lx:,1+shift+dim] # 4
    global_dim_3 = xs[lx:,2+shift+dim] # 5

# Sort with respect to ground truth
# data = zip(ground_dim,pred_dim_1, ground_dim_2, ground_dim_3)
# data = sorted(data, key=lambda tup: tup[0])
# ground_dim_sort, pred_dim_sort_1, ground_dim_sort_2, ground_dim_sort_3 = zip(*data)

# sorts all three dimenions for YPR
data = zip(ground_dim_1,pred_dim_1, global_dim_1)
data = sorted(data, key=lambda tup: tup[0])
ground_dim_sort_1, pred_dim_sort_1, global_dim_sort_1 = zip(*data)

data = zip(ground_dim_2,pred_dim_2, global_dim_2)
data = sorted(data, key=lambda tup: tup[0])
ground_dim_sort_2, pred_dim_sort_2, global_dim_sort_2 = zip(*data)

data = zip(ground_dim_3,pred_dim_3, global_dim_3)
data = sorted(data, key=lambda tup: tup[0])
ground_dim_sort_3, pred_dim_sort_3, global_dim_sort_3 = zip(*data)


font = {'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2.5)

# plt.tight_layout()

with sns.axes_style("darkgrid"):
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)


my_dpi = 300
plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
# ax1.set_title('Model Prediction vs Ground Truth Dynamics Step')
# ax11 = ax1.twinx()
# ax11.plot(global_dim_sort_1, label='Global State Variable', color='r', linestyle=':', linewidth=.8)#, linestyle=':')
ax1.axhline(0, linestyle=':', color ='r', linewidth=1)
ax1.plot(ground_dim_sort_1, label='Ground Truth', color='k', linewidth=1.8)
ax1.plot(pred_dim_sort_1, ':', label='Model Prediction', markersize=.9, linewidth=.8)##, linestyle=':')
# ax1.set_xlabel('Sorted Datapoints')
ax1.set_ylabel('Pitch Step (Deg.)')
ax1.set_ylim([-4,4])
ax1.set_yticks(np.arange(-4,4.01,2))

# ax1.legend()
# plt.show()

# plt.title('One Step Dim+1')
# ax22 = ax2.twinx()
# ax22.plot(global_dim_sort_2, label='Global State Variable', color='r', linestyle=':', linewidth=.8)#, linestyle=':')
ax2.axhline(0, linestyle=':', color ='r', linewidth=1)
ax2.plot(ground_dim_sort_2, label='Ground Truth', color='k', linewidth=1.8)
ax2.plot(pred_dim_sort_2, ':', label='Model Prediction',  markersize=.9,linewidth=.8)##, linestyle=':')

# ax2.set_xlabel('Sorted Datapoints')
ax2.set_ylabel('Roll Step (Deg.)')
ax2.set_ylim([-4,4])
ax2.set_yticks(np.arange(-4,4.01,2))
# ax2.set_yticklabels(["-5", "-2.5", "0", "2.5", "5"])

# ax2.legend()
# plt.show()

# plt.title('One Step Dim+2')
# ax33 = ax3.twinx()
# ax33.plot(global_dim_sort_3, label='Global State Variable', color='r', linestyle=':', linewidth=.8)#, linestyle=':')
ax3.axhline(0, linestyle=':', color ='r', linewidth=1)
ax3.plot(ground_dim_sort_3, label='Ground Truth', color='k', linewidth=1.8)
ax3.plot(pred_dim_sort_3, ':', label='Model Prediction', markersize=.9, linewidth=.8)#, linestyle=':')

ax3.set_xlabel('Sorted Datapoints')
ax3.set_ylabel('Yaw Step (Deg.)')
ax3.set_ylim([-4,4])
ax3.set_yticks(np.arange(-4,4.01,2))
leg3 = ax3.legend(loc=8, ncol=2)
for line in leg3.get_lines():
    line.set_linewidth(2.5)
plt.show()
# plt.savefig('_results/tempfig')
