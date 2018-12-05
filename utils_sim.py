# Our infrastucture files
from utils_data import *

# data packages
import pickle
import random

# neural nets
from model_general_nn import *
from model_split_nn import SplitModel
from _activation_swish import Swish
from model_ensemble_nn import EnsembleNN

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

def get_action(cur_state, model, method = 'Random'):
    '''Returns an action for the robot given the current state and the model'''
    print("NOT DONE")

def plot_traj_model(df_traj, model):
    # plots all the states predictions over time

    state_list, input_list, target_list = model.get_training_lists()
    data_params = {
        'states' : state_list,
        'inputs' : input_list,
        'targets' : target_list,
        'battery' : True
    }

    X, U, dX = df_to_training(df_traj, data_params)

    num_skip = 0
    X, U, dX = X[num_skip:,:], U[num_skip:,:], dX[num_skip:,:]
    # Gets starting state
    x0 = X[0,:]

    # get dims
    stack = int((len(X[0,:]))/9)
    xdim = 9
    udim = 4

    # store values
    pts = len(df_traj)-num_skip
    x_stored = np.zeros((pts, stack*xdim))
    x_stored[0,:] = x0
    x_shift = np.zeros(len(x0))

    ####################### Generate Data #######################
    for t in range(pts-1):
        # predict
        # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], U[t,:])
        x_pred = predict_nn_v2(model, x_stored[t,:], U[t,:])

        if stack > 1:
            # shift values
            x_shift[:9] = x_pred
            x_shift[9:-1] = x_stored[t,:-10]
        else:
            x_shift = x_pred

        # store values
        x_stored[t+1,:] = x_shift

    ####################### PLOT #######################
    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(331)
        ax2 = plt.subplot(332)
        ax3 = plt.subplot(333)
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)
        ax9 = plt.subplot(339)

    plt.title("Comparing Dynamics Model to Ground Truth")

    ax1.set_ylim([-150,150])
    ax2.set_ylim([-150,150])
    ax3.set_ylim([-150,150])
    ax4.set_ylim([-35,35])
    ax5.set_ylim([-35,35])
    # ax6.set_ylim([-35,35])
    ax7.set_ylim([-6,6])
    ax8.set_ylim([-6,6])
    ax9.set_ylim([5,15])

    ax1.plot(x_stored[:,0], linestyle = '--', color='b', label ='Predicted')
    ax1.plot(X[:,0], color = 'k', label = 'Ground Truth')

    ax2.plot(x_stored[:,1], linestyle = '--', color='b', label ='Predicted')
    ax2.plot(X[:,1], color = 'k', label = 'Ground Truth')

    ax3.plot(x_stored[:,2], linestyle = '--', color='b', label ='Predicted')
    ax3.plot(X[:,2], color = 'k', label = 'Ground Truth')

    ax4.plot(x_stored[:,3], linestyle = '--', color='b', label ='Predicted')
    ax4.plot(X[:,3], color = 'k', label = 'Ground Truth')

    ax5.plot(x_stored[:,4], linestyle = '--', color='b', label ='Predicted')
    ax5.plot(X[:,4], color = 'k', label = 'Ground Truth')

    ax6.plot(x_stored[:,5], linestyle = '--', color='b', label ='Predicted')
    ax6.plot(X[:,5], color = 'k', label = 'Ground Truth')

    ax7.plot(x_stored[:,6], linestyle = '--', color='b', label ='Predicted')
    ax7.plot(X[:,6], color = 'k', label = 'Ground Truth')

    ax8.plot(x_stored[:,7], linestyle = '--', color='b', label ='Predicted')
    ax8.plot(X[:,7], color = 'k', label = 'Ground Truth')

    ax9.plot(x_stored[:,8], linestyle = '--', color='b', label ='Predicted')
    ax9.plot(X[:,8], color = 'k', label = 'Ground Truth')

    ax1.legend()
    # ax2.plot(X[point:point+T+1,3:5])
    plt.show()

def plot_battery_thrust(df_traj, model):
    '''
    Function that will display a plot of the battery voltage verses motor thrust, for the appendix of the papel
    '''
    state_list, input_list, target_list = model.get_training_lists()
    data_params = {
        'states': state_list,
        'inputs': input_list,
        'targets': target_list,
        'battery': True
    }

    if 'vbat' not in input_list:
        raise ValueError("Did not include battery voltage for battery plotting")

    X, U, dX = df_to_training(df_traj, data_params)

    # plot properties
    font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # num_skip = 0
    # X, U, dX = X[num_skip:, :], U[num_skip:, :], dX[num_skip:, :]
    # # Gets starting state
    # x0 = X[0, :]

    # # get dims
    # stack = int((len(X[0, :]))/9)
    # xdim = 9
    # udim = 4

    # # store values
    # pts = len(df_traj)-num_skip
    # x_stored = np.zeros((pts, stack*xdim))
    # x_stored[0, :] = x0
    # x_shift = np.zeros(len(x0))

    thrust = np.mean(U[:,:4],axis=1)
    vbat =  U[:,-1]

    ####################### PLOT #######################
    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(111)
        # ax2 = plt.subplot(212)

    plt.title("Comparing Battery Voltage to Thrust")

    # ax1.set_ylim([-150, 150])
    # ax2.set_ylim([-150, 150])
    time = np.linspace(0,len(thrust)*.02,len(thrust))
    ln1 = ax1.plot(time, thrust, color ='r', label='Crazyflie Thrust')
    ax1.set_ylabel("Crazyflie Thrust (PWM)")

    ax2 = ax1.twinx()
    ln2 = ax2.plot(time, vbat, label='Crazyflie Logged Battery Voltage')
    ax2.set_ylabel("Crazyflie Battery Voltage (mV)")
    ax1.set_xlabel("Time (s)")
    
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    plt.show()


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

def plot_voltage_context(model, df, action = [37000,37000, 30000, 45000], act_range = 25000, normalize = False, ground_truth = False, model_nobat = []):
    '''
    Takes in a dynamics model and plots the distributions of points in the dataset
      and plots various lines verses different voltage levels
    '''

    ################ Figure out what to do with the dataframe ################
    if 'vbat' not in df.columns.values:
        raise ValueError("This function requires battery voltage in the loaded dataframe for contextual plotting")

    ################# Make sure the model is in eval mode ################
    model.eval()

    ################### Take the specific action rnage! #####################
    # going to need to set a specific range of actions that we are looking at.

    print("Looking around the action of: ", action, "\n    for a range of: ", act_range)

    # grab unique actions
    pwms_vals = np.unique(df[['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0']].values)


    # grabs the actions within the range for each motor
    pwms_vals_range1 = pwms_vals[(pwms_vals < action[0]+act_range) & (pwms_vals > action[0]-act_range)]
    pwms_vals_range2 = pwms_vals[(pwms_vals < action[1]+act_range) & (pwms_vals > action[1]-act_range)]
    pwms_vals_range3 = pwms_vals[(pwms_vals < action[2]+act_range) & (pwms_vals > action[2]-act_range)]
    pwms_vals_range4 = pwms_vals[(pwms_vals < action[3]+act_range) & (pwms_vals > action[3]-act_range)]

    # filters the dataframe by these new conditions
    df_action_filtered = df.loc[(df['m1_pwm_0'].isin(pwms_vals_range1) &
                                 df['m2_pwm_0'].isin(pwms_vals_range2) &
                                 df['m3_pwm_0'].isin(pwms_vals_range3) &
                                 df['m4_pwm_0'].isin(pwms_vals_range4))]

    if len(df_action_filtered) == 0:
        raise ValueError("Given action not present in dataset")

    if len(df_action_filtered) < 10:
        print("WARNING: Low data for this action (<10 points)")

    print("Number of datapoints found is: ", len(df_action_filtered))


    ######################## batch data by rounding voltages ################
    df = df_action_filtered.sort_values('vbat')
    # df = df_action_filtered

    num_pts = len(df)

    # spacing = np.linspace(0,num_pts,num_ranges+1, dtype=np.int)

    # parameters can be changed if desired
    state_list, input_list, change_list = model.get_training_lists()

    # For this function append vbat if not in
    v_in_flag = True
    if 'vbat' not in input_list:
        v_in_flag = False
        input_list.append('vbat')


    data_params = {
        # Note the order of these matters. that is the order your array will be in
        'states' : state_list,

        'inputs' : input_list,

        'targets' : change_list,

        'battery' : True                    # Need to include battery here too
    }

    # this will hold predictions and the current state for ease of plotting
    predictions = np.zeros((num_pts, 2*9+1))

    X, U, dX = df_to_training(df, data_params)


    # gather predictions
    rmse = np.zeros((9))
    for n, (x, u, dx) in enumerate(zip(X, U, dX)):
        # predictions[i, n, 9:] = x[:9]+model.predict(x,u)
        if ground_truth:
            predictions[n, 9:-1] = dx
        else:
            # hacky solution to comparing models tranined with and without battery
            if v_in_flag:
                predictions[n, 9:-1] = model.predict(x,u)
            else:
                predictions[n, 9:-1] = model.predict(x,u[:-1])

            # calculate root mean squared error for predictions
            rmse += (predictions[n, 9:-1] - dx)**2

        predictions[n, :9] = x[:9]     # stores for easily separating generations from plotting
        predictions[n, -1] = u[-1]

    rmse /= n
    rmse = np.sqrt(rmse)
    print(rmse)


    # if normalize, normalizes both the raw states and the change in states by
    #    the scalars stored in the model
    if normalize:
        scalarX, scalarU, scalardX = model.getNormScalers()
        prediction_holder = np.concatenate((predictions[:,:9],np.zeros((num_pts, (np.shape(X)[1]-9)))),axis=1)
        predictions[:,:9] = scalarX.transform(prediction_holder)[:,:9]
        predictions[:,9:-1] = scalardX.transform(predictions[:,9:-1])

    ############################################################################
    ############################################################################
    ######################### plot this dataset on Euler angles ################
    # this will a subplot with a collection of points showing the next state
    #   that originates from a initial state. The different battery voltages will
    #   be different colors. They could be lines, but is easier to thing about
    #   in an (x,y) case without sorting

    # plot properties
    font = {'size'   : 14}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    ############## PLOT ALL POINTS ON 3 EULER ANGLES ###################
    if False:
        with sns.axes_style("darkgrid"):
            fig1, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
            ax1, ax2, ax3 = axes[:]

            if ground_truth:
                plt.suptitle("Measured State Transitions Battery Voltage Context - Action: {0}".format(action))
                if normalize:
                    ax1.set_ylabel("Measured Normalized Change in State")
                else:
                    ax1.set_ylabel("Measured Change in state (Degrees)")
            else:
                plt.suptitle("Predicted State Transitions Battery Voltage Context - Action: {0}".format(action))
                if normalize:
                    ax1.set_ylabel("Predicted Normalized Change in State")
                else:
                    ax1.set_ylabel("Predicted Change in state (Degrees)")

            ax1.set_title("Pitch")
            ax2.set_title("Roll")
            ax3.set_title("Yaw")

            if normalize:
                ax1.set_xlabel("Normalized Pitch")
                ax2.set_xlabel("Normalized Roll")
                ax3.set_xlabel("Normalized Yaw")
                # ax1.set_xlim([-4,4])
                # ax2.set_xlim([-4,4])
                # ax3.set_xlim([-2,2])
                # ax1.set_xlim([-1,1])
                # ax2.set_xlim([-1,1])
                # ax3.set_xlim([-2,2])
                ax1.set_ylim([-1,1])
                ax2.set_ylim([-1,1])
                ax3.set_ylim([-1,1])
            else:
                ax1.set_xlabel("Global Pitch")
                ax2.set_xlabel("Global Roll")
                ax3.set_xlabel("Global Yaw")
                ax1.set_xlim([-45,45])
                ax2.set_xlim([-45,45])
                ax3.set_xlim([-180,180])

            fig1.subplots_adjust(right=0.8)
            cbar_ax1 = fig1.add_axes([0.85, 0.15, 0.02, 0.7])
            # ax1 = plt.subplot(131)
            # ax2 = plt.subplot(132)
            # ax3 = plt.subplot(133)

        # normalize batteris between 0 and 1
        # TODO: Figure out the coloring
        # predictions[:,:,-1] = (predictions[:,:,-1] - np.min(predictions[:,:,-1]))/(np.max(predictions[:,:,-1])-np.min(predictions[:,:,-1]))
        # print(predictions[:,:,-1])
        base = 50
        prec = 0
        vbats = np.around(base * np.around(predictions[:, -1]/base),prec)
        # vbats = predicitons[:,-1]
        hm = ax1.scatter(predictions[:,3], predictions[:,3+9], c=vbats, alpha = .7, s=3)
        ax2.scatter(predictions[:,4], predictions[:,4+9], c=vbats, alpha = .7, s=3)
        ax3.scatter(predictions[:,5], predictions[:,5+9], c=vbats, alpha = .7, s=3)
        cbar = fig1.colorbar(hm, cax=cbar_ax1)
        cbar.ax.set_ylabel('Battery Voltage (mV)')

        plt.show()
        ###############################################################

    ############## PLOT Pitch for battery cutoff ###################
    if False:
        battery_cutoff = 3800
        battery_cutoff = int(np.mean(predictions[:, -1]))
        battery_cutoff = int(np.median(predictions[:, -1]))

        print("Plotting Pitch Dynamics for Above and Below {0} mV".format(battery_cutoff))
        with sns.axes_style("darkgrid"):
            fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharey=True)
            ax21, ax22 = axes2[:]

            cmap = matplotlib.cm.viridis
            norm = matplotlib.colors.Normalize(vmin=np.min(predictions[:, -1]), vmax=np.max(predictions[:, -1]))

            if ground_truth:
                plt.suptitle("Measured Pitch Transitions Above and Below Mean Vbat: {0}".format(battery_cutoff))
                if normalize:
                    ax21.set_ylabel("Normalized Measured Change in State")
                else:
                    ax21.set_ylabel("Measured Change in state (Degrees)")
            else:
                plt.suptitle("Predicted Pitch Transitions Above and Below Mean Vbat: {0}".format(battery_cutoff))
                if normalize:
                    ax21.set_ylabel("Normalized Predicted Change in State")
                else:
                    ax21.set_ylabel("Predicted Change in state (Degrees)")

            ax21.set_title("Pitch, Vbat > {0}".format(battery_cutoff))
            ax22.set_title("Pitch, Vbat < {0}".format(battery_cutoff))

            if normalize:
                ax21.set_xlabel("Normalized Pitch")
                ax22.set_xlabel("Normalized Pitch")
                # ax21.set_xlim([-4,4])
                # ax22.set_xlim([-4,4])
                ax21.set_ylim([-1,1])
                ax22.set_ylim([-1,1])
            else:
                ax21.set_xlabel("Global Pitch")
                ax22.set_xlabel("Global Pitch")
                ax21.set_xlim([-45,45])
                ax22.set_xlim([-45,45])

            fig2.subplots_adjust(right=0.8)
            cbar_ax = fig2.add_axes([0.85, 0.15, 0.02, 0.7])

        dim = 3
        base = 50
        prec = 1
        vbats = np.around(base * np.around(predictions[:, -1]/base),prec)
        flag = vbats > battery_cutoff
        notflag = np.invert(flag)
        # hm2 = plt.scatter(predictions[:,3], predictions[:,3+9], c=predictions[:, -1], alpha = .7, s=3)
        # plt.clf()
        ax21.scatter(predictions[flag, dim], predictions[flag, dim+9], cmap=cmap, norm=norm, c=vbats[flag], alpha = .7, s=3)
        ax22.scatter(predictions[notflag, dim], predictions[notflag, dim+9], cmap=cmap, norm=norm, c=vbats[notflag], alpha = .7, s=3)
        cbar = fig2.colorbar(hm, cax=cbar_ax)
        cbar.ax.set_ylabel('Battery Voltage (mV)')

        plt.show()
        ###############################################################

    if False:
        num_subplots = 9
        vbats = predictions[:, -1]

        # generate battery ranges for the plot
        pts = len(vbats)
        pts_breaks = np.linspace(0,pts-1, num_subplots+1, dtype =np.int)
        bat_ranges = vbats[pts_breaks]


        # bat_ranges = np.linspace(np.min(vbats), np.max(vbats),num_subplots+1)

        with sns.axes_style("darkgrid"):
            fig3, axes3 = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True)
            # ax31, ax32, ax33, ax34, ax35, ax36 = axes3[:,:]

            cmap = matplotlib.cm.viridis
            norm = matplotlib.colors.Normalize(vmin=bat_ranges[0], vmax=bat_ranges[-1])

            if ground_truth:
                plt.suptitle("Measured Pitch Transitions For Varying Battery Voltage")
                if normalize:
                    fig3.text(0.5, 0.04, 'Normalize Global State', ha='center')
                    fig3.text(0.04, 0.5, 'Normalized Measured Change in State', va='center', rotation='vertical')
                else:
                    fig3.text(0.5, 0.04, 'Global State', ha='center')
                    fig3.text(0.04, 0.5, 'Measured Change in State', va='center', rotation='vertical')
            else:
                plt.suptitle("Predicted Pitch Transitions For Varying Battery Voltage")
                if normalize:
                    fig3.text(0.5, 0.04, 'Normalize Global State', ha='center')
                    fig3.text(0.04, 0.5, 'Normalized Predicted Change in State', va='center', rotation='vertical')
                else:
                    fig3.text(0.5, 0.04, 'Global State', ha='center')
                    fig3.text(0.04, 0.5, 'Predicted Change in State', va='center', rotation='vertical')


            for i, ax in enumerate(axes3.flatten()):
                # get range values
                low = bat_ranges[i]
                high = bat_ranges[i+1]


                ax.set_title("Voltage [{0},{1}]".format(int(low), int(high)))
                if normalize:
                    # ax.set_xlabel("Normalized Pitch")
                    ax.set_ylim([-1,1])
                else:
                    # ax.set_xlabel("Global Pitch")
                    ax.set_xlim([-45,45])

                dim = 4
                flag = (vbats > low) & (vbats < high)
                hm = ax.scatter(predictions[flag, dim], predictions[flag, dim+9], cmap = cmap, norm = norm, c=vbats[flag], alpha = .7, s=3)

                if normalize:
                    ax.set_ylim([-1,1])
                else:
                    ax.set_ylim([-3,3])

            fig3.subplots_adjust(right=0.8)
            cbar_ax1 = fig3.add_axes([0.85, 0.15, 0.02, 0.7])
            cbar = fig3.colorbar(hm, cax=cbar_ax1)
            cbar.ax.set_ylabel('Battery Voltage (mV)')

            plt.show()
            ###############################################################

    ############## PLOT single angle for ground truth, with battery, without battery ###################
    if True:

        # gather predictions for second model
        # this will hold predictions and the current state for ease of plotting
        predictions_nobat = np.zeros((num_pts, 2*9+1))
        pred_ground_truth = np.zeros((num_pts, 2*9+1))

        # gather predictions
        rmse = np.zeros((9))
        for n, (x, u, dx) in enumerate(zip(X, U, dX)):
            # predictions[i, n, 9:] = x[:9]+model.predict(x,u)
            pred_ground_truth[n, 9:-1] = dx
            predictions_nobat[n, 9:-1] = model_nobat.predict(x, u[:-1])

            # calculate root mean squared error for predictions
            rmse += (predictions_nobat[n, 9:-1] - dx)**2

            # stores for easily separating generations from plotting
            predictions_nobat[n, :9] = x[:9]
            predictions_nobat[n, -1] = u[-1]

        # rmse /= n
        # rmse = np.sqrt(rmse)
        # print(rmse)

        if normalize:
            scalarX, scalarU, scalardX = model.getNormScalers()
            pred_ground_truth_holder = np.concatenate(
                (pred_ground_truth[:, :9], np.zeros((num_pts, (np.shape(X)[1]-9)))), axis=1)
            pred_ground_truth[:, :9] = scalarX.transform(
                pred_ground_truth_holder)[:, :9]
            pred_ground_truth[:, 9:-
                              1] = scalardX.transform(pred_ground_truth[:, 9:-1])

            prediction_nobat_holder = np.concatenate(
                (predictions_nobat[:, :9], np.zeros((num_pts, (np.shape(X)[1]-9)))), axis=1)
            predictions_nobat[:, :9] = scalarX.transform(
                prediction_nobat_holder)[:, :9]
            predictions_nobat[:, 9:-
                              1] = scalardX.transform(predictions_nobat[:, 9:-1])


        # Plot here, will be a 3x5 plot of voltage context
        n_row = 3
        num_subplots = 5
        vbats = predictions[:, -1]

        # generate battery ranges for the plot
        pts = len(vbats)
        pts_breaks = np.linspace(0, pts-1, num_subplots+1, dtype=np.int)
        bat_ranges = vbats[pts_breaks]

        # bat_ranges = np.linspace(np.min(vbats), np.max(vbats),num_subplots+1)

        with sns.axes_style("darkgrid"):
            fig3, axes3 = plt.subplots(nrows=n_row, ncols=num_subplots, sharey=True, sharex=True)
            # ax31, ax32, ax33, ax34, ax35, ax36 = axes3[:,:]

            plt.suptitle("Voltage Context Effect on Prediction")
            fig3.text(0.45, 0.01, 'Pitch (Degrees)', ha='center')


            cmap = matplotlib.cm.viridis
            norm = matplotlib.colors.Normalize(vmin=bat_ranges[0], vmax=bat_ranges[-1])


            for i, ax in enumerate(axes3.flatten()):

                if (i % 5 == 0):
                    if i < num_subplots:
                        ax.set_ylabel("Ground Truth Changes")
                    elif i < 2*num_subplots:
                        ax.set_ylabel("Predicted with Battery")
                    else:
                        ax.set_ylabel("Predicted  without Battery")

                j = (i % num_subplots)
                # get range values
                low = bat_ranges[j]
                high = bat_ranges[j+1]
                
                if i < num_subplots: 
                    ax.set_title("Voltage [{0},{1}]".format(int(low), int(high)))
                    
                if normalize:
                    if i < num_subplots:
                        ax.set_xlabel("Normalized Pitch")
                    ax.set_ylim([-1, 1])
                else:
                    if i < num_subplots:
                        ax.set_xlabel("Global Pitch")
                    ax.set_xlim([-45, 45])

                dim = 4
                flag = (vbats > low) & (vbats < high)
                if i < num_subplots:
                    hm = ax.scatter(predictions[flag, dim], pred_ground_truth[flag, dim+9],
                                    cmap=cmap, norm=norm, c=vbats[flag], alpha=.7, s=3)
                elif i < 2* num_subplots:
                    hm = ax.scatter(predictions[flag, dim], predictions[flag, dim+9],
                                    cmap=cmap, norm=norm, c=vbats[flag], alpha=.7, s=3)
                else:
                    hm = ax.scatter(predictions[flag, dim], predictions_nobat[flag, dim+9],
                                    cmap=cmap, norm=norm, c=vbats[flag], alpha=.7, s=3)
                # if normalize:
                #     ax.set_ylim([-1, 1])
                # else:
                #     ax.set_ylim([-3, 3])

            fig3.subplots_adjust(right=0.8)
            cbar_ax1 = fig3.add_axes([0.85, 0.15, 0.02, 0.7])
            cbar = fig3.colorbar(hm, cax=cbar_ax1)
            cbar.ax.set_ylabel('Battery Voltage (mV)')

            plt.show()
            ###############################################################


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
        x_noise_vec = np.random.normal(
            loc=0, scale=self.x_noise, size=(self.x_dim))

        # makes states less than 1e-12 = 0
        x1[x1 < 1e-12] = 0
        return x1+x_noise_vec
