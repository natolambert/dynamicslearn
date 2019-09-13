# Our infrastucture files
from utils.data import *
from utils.nn import *
from torch.utils.data import Dataset, DataLoader
# from gymenv_quad import QuadEnv

# data packages
import pickle
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

# # neural nets
# from model_split_nn import SplitModel
# from _activation_swish import Swish
# from model_ensemble_nn import EnsembleNN

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os
import copy

# Plotting
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit


def explore_pwm_equil(df):
    """
    Function that takes in a dataset and a model and will look through the distributions of PWM actions
      for which the change in angles was low to try and derive a psuedo equilibrium for a given dataset.
    """

    # Note on dimensions
    # 0  1  2  3  4  5  6  7  8
    # wx wy wz p  r  y  lx ly lz
    def gaussian(x, amp, cen, wid):
        return amp * exp(-(x-cen)**2 / wid)

    conditions = {
        'objective vals': 0,
        'd_roll': .05,
        'd_pitch': .05,
        'd_yaw': .05,
        'pitch0': 5,
        'roll0': 5
    }

    # fit gaussian:
    # xdata - 
    # ydata - 
    # popt pcov = curve_fit(gaussian, xdata, ydata)

    # generate mean and variance of the PWMs
    for cond in conditions.items():
        var = cond[0]
        tolerance = cond[1]
        # print(var)
        # print(-tolerance, tolerance)
        if var == 'objective vals':
            df = df.loc[df['objective vals']>0]
        else:
            df = df.loc[df[var].between(-tolerance, tolerance)]

    # print(df)
    df_actions = df[['m1_pwm_0','m2_pwm_0','m3_pwm_0','m4_pwm_0']]
    # print(df_actions)

    print('----')
    print("Number of points in this estimate: ", len(df_actions))
    print("Equil actions: ",np.mean(df_actions.values, axis=0))
    print("Std Dev:", np.std(df_actions.values, axis=0))
    

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    rounded_act = np.zeros((4))
    for i in range(4):
        rounded_act[i] = find_nearest(
            df_actions.iloc[i], np.mean(df_actions.values, axis=0)[i])

    print("Most Common Action in data: ", rounded_act)
    print('----')
    return rounded_act



def pred_traj(x0, action, model, T):
    # get dims
    stack = int((len(x0))/9)
    xdim = 9
    udim = 4

    state_list, input_list, target_list = model.get_training_lists()


    # figure out if given an action or a controller
    if not isinstance(action, np.ndarray):
        # given PID controller. Generate actions as it goes
        mode = 1

        PID = copy.deepcopy(action) # for easier naming and resuing code

        # create initial action
        action_eq = np.array([30687.1, 33954.7, 34384.8, 36220.11]) #[31687.1, 37954.7, 33384.8, 36220.11])
        action = np.array([30687.1, 33954.7, 34384.8, 36220.11])
        if stack > 1:
            action = np.tile(action, stack)
        if 'vbat' in input_list:
            action = np.concatenate((action,[3900]))

        # step 0 PID response
        action[:udim] += PID.update(x0[4])
    else:
        mode = 0

    # function to generate trajectories
    x_stored = np.zeros((T+1,len(x0)))
    x_stored[0,:] = x0
    x_shift = np.zeros(len(x0))

    for t in range(T):
        if mode == 1:
            # predict with actions coming from controller
            if stack > 1:       # if passed array of actions, iterate
                # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)
                x_pred = predict_nn_v2(model, x_stored[t,:], action)
                # slide action here
                action[udim:-1] = action[:-udim-1]

            else:
                # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)
                x_pred = predict_nn_v2(model, x_stored[t,:], action)

            # update action
            PIDout = PID.update(x_pred[4])
            action[:udim] = action_eq+np.array([1,1,-1,-1])*PIDout
            print("=== Timestep: ", t)
            print("Predicted angle: ", x_pred[4])
            print("PIDoutput: ", PIDout)
            print("Given Action: ", action[:udim])

        # else give action array
        elif mode == 0:
            # predict
            if stack > 1:       # if passed array of actions, iterate
                # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action[t,:])
                x_pred = predict_nn_v2(model, x_stored[t,:], action[t,:])
            else:
                # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)
                x_pred = predict_nn_v2(model, x_stored[t,:], action)

        # shift values
        x_shift[:9] = x_pred
        x_shift[9:-1] = x_stored[t,:-10]

        # store values
        x_stored[t+1,:] = x_shift

    x_stored[:,-1] = x0[-1]     # store battery for all (assume doesnt change on this time horizon)

    return x_stored

def gather_predictions(model_dir, dataset, delta=True, variances=False):
    """
    Takes in a dataset and returns a matrix of predictions for plotting.
    - model_dir of the form '_models/temp/... .pth'
    - dataset of the form (X, U, dX)
    - delta makes the plot of the change in state or global predictions 
    - note that predict_nn_v2 returns the global values, always
    """

    nn = torch.load(model_dir)
    # nn.training = False
    nn.eval()
    with open(model_dir[:-4]+'--normparams.pkl', 'rb') as pickle_file:
        normX1, normU1, normdX1 = pickle.load(pickle_file)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    # note which states are raw vs full change in state
    pred_key = nn.change_state_list
    pred_key_bool = [p[:2] == 'd_' for p in pred_key]


    # Original gather predictions
    if not variances:
        predictions_1 = np.empty((0, 9))  # np.shape(X)[1]))
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

    # new gather predictions. Else we return variance predictions too
    else:
        predictions_1 = np.empty((0, 9))  # np.shape(X)[1]))
        predictions_1_var = np.empty((0, 9))  # np.shape(X)[1]))
        # predictions_1 = torch.Tensor([[0,0,0,0,0,0,0,0,0]])
        # predictions_1_var = torch.Tensor([[0,0,0,0,0,0,0,0,0]])
        for (dx, x, u) in zip(dX, X, U):
            # pred = predict_nn_v2(nn, x, u)

            
            x = torch.Tensor(x)
            u = torch.Tensor(u)
            # grab prediction value
            mean, var = nn.distribution(x, u)

            # for i, b in enumerate(pred_key_bool):
            #     if b:
            #         mean[i] = x[i] + mean[i]
                # else:
                    # mean[i] = mean[i]
            print("MEAN")
            print(mean.shape)
            print(predictions_1.shape)
            predictions_1 = np.append(
                predictions_1, mean.reshape(1, -1).detach().numpy(),  axis=0)
            predictions_1_var = np.append(
                predictions_1_var, var.reshape(1, -1).detach().numpy(),  axis=0)
            # predictions_1 = np.append(
            #     predictions_1, mean.detach().numpy(),  axis=0)
            # predictions_1_var = np.append(
            #     predictions_1_var, var.detach().numpy(),  axis=0)
            
        print(predictions_1[1:, :].shape)
        return predictions_1, predictions_1_var
        # return predictions_1[1:, :], predictions_1_var[1:, :]

class CrazyFlie():
    def __init__(self, dt, m=.035, L=.065, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, x_noise=.0001, u_noise=0):
        _state_dict = {
            'X': [0, 'pos'],
            'Y': [1, 'pos'],
            'Z': [2, 'pos'],
            'vx': [3, 'vel'],
            'vy': [4, 'vel'],
            'vz': [5, 'vel'],
            'yaw': [6, 'angle'],
            'pitch': [7, 'angle'],
            'roll': [8, 'angle'],
            'w_x': [9, 'omega'],
            'w_y': [10, 'omega'],
            'w_z': [11, 'omega']
        }
        # user can pass a list of items they want to train on in the neural net, eg learn_list = ['vx', 'vy', 'vz', 'yaw'] and iterate through with this dictionary to easily stack data

        # input dictionary less likely to be used because one will not likely do control without a type of acutation. Could be interesting though
        _input_dict = {
            'Thrust': [0, 'force'],
            'taux': [1, 'torque'],
            'tauy': [2, 'torque'],
            'tauz': [3, 'torque']
        }
        self.x_dim =12
        self.u_dim = 4
        self.dt = dt

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        # Setup the parameters
        self.m = m
        self.L = L
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = 9.81

        # Define equilibrium input for quadrotor around hover
        self.u_e = np.array([m*self.g, 0, 0, 0])               #This is not the case for PWM inputs
        # Four PWM inputs around hover, extracted from mean of clean_hover_data.csv
        # self.u_e = np.array([42646, 40844, 47351, 40116])

        # Hover control matrices
        self._hover_mats = [np.array([1, 0, 0, 0]),      # z
                            np.array([0, 1, 0, 0]),   # pitch
                            np.array([0, 0, 1, 0])]   # roll

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(
                                    x0[0]),                   -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
        return rotn_matrix.dot(pqr)

    def pwm_thrust_torque(self, PWM):
        # Takes in the a 4 dimensional PWM vector and returns a vector of 
        # [Thrust, Taux, Tauy, Tauz] which is used for simulating rigid body dynam
        # Sources of the fit: https://wiki.bitcraze.io/misc:investigations:thrust, 
        #   http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8905295&fileOId=8905299

        # The quadrotor is 92x92x29 mm (motor to motor, square along with the built in prongs). The the distance from the centerline, 
        
        # Thrust T = .35*d + .26*d^2 kg m/s^2 (d = PWM/65535 - normalized PWM)
        # T = (.409e-3*pwm^2 + 140.5e-3*pwm - .099)*9.81/1000 (pwm in 0,255)

        def pwm_to_thrust(PWM):
            # returns thrust from PWM
            pwm_n = PWM/65535.0
            thrust = .35*pwm_n + .26*pwm_n**2
            return thrust

        pwm_n = np.sum(PWM)/(4*65535.0)

        l = 35.527e-3   # length to motors / axis of rotation for xy
        lz = 46         # axis for tauz
        c = .05         # coupling coefficient for yaw torque

        # Torques are slightly more tricky
        # x = m2+m3-m1-m4
        # y =m1+m2-m3-m4
    
        # Estiamtes forces
        m1 = pwm_to_thrust(PWM[0])
        m2 = pwm_to_thrust(PWM[1])
        m3 = pwm_to_thrust(PWM[2])
        m4 = pwm_to_thrust(PWM[3])

        Thrust = pwm_to_thrust(np.sum(PWM)/(4*65535.0))
        taux = l*(m2+m3-m4-m1)
        tauy = l*(m1+m2-m3-m4)
        tauz = -lz*c*(m1+m3-m2-m4)

        return np.array([Thrust, taux, tauy, tauz])

    def simulate(self, x, PWM, t=None):
        # Input structure:
        # u1 = thrust
        # u2 = torque-wx
        # u3 = torque-wy
        # u4 = torque-wz
        u = self.pwm_thrust_torque(PWM)
        dt = self.dt
        u0 = u
        x0 = x
        idx_xyz = self.idx_xyz
        idx_xyz_dot = self.idx_xyz_dot
        idx_ptp = self.idx_ptp
        idx_ptp_dot = self.idx_ptp_dot

        m = self.m
        L = self.L
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        g = self.g

        Tx = np.array([Iyy / Ixx - Izz / Ixx, L / Ixx])
        Ty = np.array([Izz / Iyy - Ixx / Iyy, L / Iyy])
        Tz = np.array([Ixx / Izz - Iyy / Izz, 1. / Izz])

        # # Add noise to input
        # u_noise_vec = np.random.normal(
        #     loc=0, scale=self.u_noise, size=(self.u_dim))
        # u = u+u_noise_vec

        # Array containing the forces
        Fxyz = np.zeros(3)
        Fxyz[0] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.cos(
            x0[idx_ptp[2]]) + math.sin(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[1] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.sin(
            x0[idx_ptp[2]]) - math.sin(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[2] = g - 1 * (math.cos(x0[idx_ptp[0]]) *
                           math.cos(x0[idx_ptp[1]])) * u0[0] / m

        # Compute the torques
        t0 = np.array([x0[idx_ptp_dot[1]] * x0[idx_ptp_dot[2]], u0[1]])
        t1 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[2]], u0[2]])
        t2 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[1]], u0[3]])
        Txyz = np.array([Tx.dot(t0), Ty.dot(t1), Tz.dot(t2)])

        x1 = np.zeros(12)
        x1[idx_xyz_dot] = x0[idx_xyz_dot] + dt * Fxyz
        x1[idx_ptp_dot] = x0[idx_ptp_dot] + dt * Txyz
        x1[idx_xyz] = x0[idx_xyz] + dt * x0[idx_xyz_dot]
        x1[idx_ptp] = x0[idx_ptp] + dt * \
            self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        # x_noise_vec = np.random.normal(
        #     loc=0, scale=self.x_noise, size=(self.x_dim))

        # makes states less than 1e-12 = 0
        x1[x1 < 1e-12] = 0
        return x1+x_noise_vec

def explore_policy():#policy_imitate, policy_sac = [], policy_mpc = False):
    '''
    Function to take a trained control policy and export a analysis 
      of what the policy will do on the trained dataset
    '''
    print("TEST")
    """ 
    Some ideas
    - can we plot distribution of actions taken over the trained dataset.
        - we could have a "validation" portion of the dataset that corresponds
            to the test data for the neural network
    - How do you test train / validate this etc
    
    Some thoughts from email:
    - perturb states slightly and check how much policies change
    - somehow plot action choices over the logged state space data and 
        see if distributions have different variance etc
    - With the action space being 4 dimensional, maybe we can reduce it to something 
        like “Thrust, roll power, and pitch power” which we could visualize in 3d
    """

    data_file = open(
        "_models/temp/2019-03-22--09-29-48.4_mfrl_ens_stack3_--data.pkl", 'rb')
    df = pickle.load(data_file)

    Env = QuadEnv()
    policy_sac = []
    policy_Impc = []
    state_data = df[['omega_x0', 'omega_y0', 'omega_z0', 'pitch0', 'roll0',
                     'yaw0', 'lina_x0', 'lina_y0', 'lina_z0', 
                     'omega_x1', 'omega_y1', 'omega_z1', 'pitch1', 'roll1',
                     'yaw1', 'lina_x1', 'lina_y1', 'lina_z1', 
                     'omega_x1', 'omega_y1', 'omega_z1', 'pitch1', 'roll1',
                     'yaw1', 'lina_x1', 'lina_y1', 'lina_z1', 
                     'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                     'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2']]

    # init action arrays
    data_mpc = df[['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0']].values
    data_sac = np.empty(np.shape(data_mpc))
    data_Impc = np.empty(np.shape(data_mpc))

    def normalize_act(action):
        # Returns 'flyable' pwm from NN output [-1,1]
        return .5*(action+1)*(Env.act_high-Env.act_low)+Env.act_low

    # itetate through all states and generate policy data
    for i,state in enumerate(state_data.values):
        # need to normalize the state for the policies
        state_norm = (state - Env.norm_means)/Env.norm_stds
        act_sac_raw, _ = policy_sac.get_action(state_norm)
        act_sac = normalize_act(act_sac_raw)
        data_sac[i, :] = act_sac

        act_Impc_raw, _ = policy_Impc.forward(state_norm)
        act_Impc = normalize_act(act_Impc_raw)
        data_Impc[i, :] = act_Impc



    quit()
