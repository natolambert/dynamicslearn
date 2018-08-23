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

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from enum import Enum

# load network
model_name = '_models/temp/2018-08-23 10/42/55.1w-150e-100lr-7e-06b-32d-2018_08_22_cf1_hover_p-True.pth'
newNN = torch.load(model_name)

normX,normU,normdX = pickle.load(model_name[:-4]+'-normparams.pkl')


# load new data


# continue training the network

 # Train
acc = newNN.train((Seqs_X, Seqs_U),
                    learning_rate = lr,
                    epochs=e,
                    batch_size = b,
                    optim="Adam")

plt.plot(acc)
plt.show()

# Saves NN params
dir_str = str('_models/temp_reinforced/')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
info_str = "||w=" + str(w) + "e=" + str(e) + "lr=" + str(lr) + "b=" + str(b) + "d=" + str(data_name) + "p=" + str(prob_flag)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')

normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"||normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)
