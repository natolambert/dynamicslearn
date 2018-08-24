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
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
import csv
from enum import Enum

# load network
model_name = '_models/current_best/2018-08-23--14-21-35.9||w=150e=250lr=7e-06b=32d=2018_08_22_cf1_hover_p=True.pth'
newNN = torch.load(model_name)

with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
    normX,normU,normdX = pickle.load(pickle_file)

new_data = []
# load new data
with open('_logged_data_autonomous/flight_log-20180823-160140.csv', "rb") as csvfile:
    new_data = np.loadtxt(csvfile, delimiter=",")


new_data = np.array(new_data)

X = new_data[:,:6]
U = new_data[:,6:10]
Time = new_data[:,10]
Objv = new_data[:,11]

# Reduces by length one for training
dX = X[1:,:]-X[:-1,:]
X = X[:-1,:]
U = U[:-1,:]
Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
Objv = Objv[:-1]
Time = Time[:-1]

# Remove data where the timestep is wrong
# Remove data if timestep above 10ms
X = X[np.array(np.where(Ts < 10)).flatten(),:]
U = U[np.array(np.where(Ts < 10)).flatten(),:]
dX = dX[np.array(np.where(Ts < 10)).flatten(),:]
Objv = Objv[np.array(np.where(Ts < 10)).flatten()]
Ts = Ts[np.array(np.where(Ts < 10)).flatten()]
Time = Time[np.array(np.where(Ts < 10)).flatten()]

# Remove data where Ts = 0
X = X[np.array(np.where(Ts != 0)).flatten(),:]
U = U[np.array(np.where(Ts != 0)).flatten(),:]
dX = dX[np.array(np.where(Ts != 0)).flatten(),:]
Objv = Objv[np.array(np.where(Ts != 0)).flatten()]
Ts = Ts[np.array(np.where(Ts != 0)).flatten()]
Time = Time[np.array(np.where(Ts != 0)).flatten()]


# remove repeated euler angles
if True:
    X = X[np.all(X[:,3:] !=0, axis=1)]
    U = U[np.all(X[:,3:] !=0, axis=1)]
    dX = dX[np.all(X[:,3:] !=0, axis=1)]


# plot
if True:
    font = {'family' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=3)

    start_time = Time[0]
    scaled_time = (Time[:]-start_time)/1000000

    ax1 = plt.subplot(311)
    plt.title('Autonomous Flight Data')

    ax1.plot(scaled_time, X[:,3:])
    ax1.set_ylim([-60,60])
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_xlabel('Time (ms)')

    # ax2 = ax1.twinx()
    ax4 = plt.subplot(312)
    ax4.plot(scaled_time, X[:,0],color='m')
    ax4.plot(scaled_time, X[:,1],color='c')
    ax4.plot(scaled_time, X[:,2],color='y')
    ax4.set_ylabel('Angular Accel (deg/s^2)')
    ax4.set_ylim([-400,400])

    ax1.legend(['pitch', 'roll', 'yaw'],loc=2)
    ax4.legend(['omega_x','omega_y','omega_z'],loc=2)

    ax3 = plt.subplot(313)
    ax3.plot(scaled_time, Objv, color='r')
    ax3.set_ylabel('Objective Function Value')
    ax3.set_ylim([0,1000])
    ax3.set_xlabel('Time (ms)')

    plt.show()
# continue training the network
print(np.shape(X))
quit()
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
