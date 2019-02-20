# File for plotting utilities

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from utils.sim import *
from utils.data import *

# Included functions with descriptions:
# plot_flight_time: Plots flight time vs rollout
# plot_trained_points: plots flight time vs log trained points
# plot_sensor_quality: plots std dev of sensor measurements vs flight chronologically
# plot_waterfall: Plots model predictions and best action for a traj
# plot_traj_model: plots model predictions from beginning of a trajectory
# plot_battery_thrust: plots thrust vs battery for a trajectory
# plot_euler_preds: plots the one step predictions for the dynamics model on euler angles
# plot_rollout_compare: Plots euler angles over time for first 4 rollouts
# plot_flight_segment: plots a segment of a flight at given location



def plot_flight_time(csv_dir):
    '''
    tool to take in a directory of flight summaries and plot the flight time vs rollouts
    the file structure should be as follows:
        flights/
          ..freq1
              ..rand
              ..roll1
              ..roll2
          ..freq2
              ..rand
              ..roll1
              ..roll2
          ....
    '''

    print_flag = False

    if print_flag:
        print('~~~~~~~~~~~~')

    # gather dirs (each will be one line on the plot)
    dirs = [dI for dI in os.listdir(
        csv_dir) if os.path.isdir(os.path.join(csv_dir, dI))]
    # print(dirs)

    font = {'size': 23}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=4.5)


    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        plt.subplots_adjust(wspace=.15, left=.07, right=1-.07)  # , hspace=.15)
        ax1 = plt.subplot(111)

    colors = ['#208080', '#F83030', '#808000']
    markers = ['*', '.', 'x']
    i = 0
    for dir, c in zip(reversed(sorted(dirs)), colors):
        if print_flag:
            print('---' + dir + '---')
        # Load each frequency fo rollouts individually
        label = dir
        files = os.listdir(csv_dir+dir)

        # create list for dataframes
        data = []
        df_main = pd.DataFrame(columns=[
                               "Flight Idx", "Flight Time (ms)", "Mean Objective", "RMS Pitch Roll", "roll"])

        means = []
        stds = []
        labels = []

        for f in sorted(files):
            if f != '.DS_Store':
                if print_flag:
                    print("-> Rollout: " + f[-10:-3])
                with open(csv_dir+dir+'/'+f, "rb") as csvfile:
                    # laod data
                    roll = f[-f[::-1].find('_'):-f[::-1].find('.')-1]
                    df = pd.read_csv(csvfile, sep=",")
                    df['roll'] = roll
                    df_main = df_main.append(df)

                    mean_sub = df["Flight Time (ms)"].mean()
                    std_sub = df["Flight Time (ms)"].std()

                    means.append(mean_sub)
                    stds.append(std_sub)
                    if roll == 'rand':
                        roll = '00'
                    else:
                        roll = roll[-2:]
                    labels.append(roll)

                    # mean = np.mean(new_data[:,1])
                    # std = np.std(new_data[:,1])

                    if print_flag:
                        new_data = np.loadtxt(
                            csvfile, delimiter=",", skiprows=1)
                        print("   Mean flight length is: ",
                              np.mean(new_data[:, 1]))
                        print("   Std flight length is: ",
                              np.std(new_data[:, 1]))

        means = np.array(means)
        stds = np.array(stds)

        x = np.arange(0, len(labels))
        ax1.plot(x, means/1000, label=label, color=c,
                 marker=markers[i], markersize='19')
        ax1.set_xlim([0,len(labels)-1])

        ax1.axhline(np.max(means)/1000, linestyle='--',
                    label=str("Max" + dir), alpha=1, color=c)

        ax1.set_ylabel("Flight Time (s)")
        ax1.set_xlabel("Rollout (10 Flights Per)")

        ax1.grid(b=True, which='major', color='k',
                 linestyle='-', linewidth=0, alpha=.5)
        ax1.grid(b=True, which='minor', color='r', linestyle='--', linewidth=0)
        # ax1.set_title("Flight Time vs Rollout")

        ax1.set_ylim([0, 2.500])

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=75, fontsize=18)

        ax1.legend()

        # , edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.fill_between(x, (means-stds)/1000, (means+stds) /
                         1000, alpha=0.3, color=c)

        i += 1
        ###############

    plt.show()

    print('\n')

def plot_trained_points(csv_dir):
    """
    tool to take in a directory of flight summaries and plot the flight points vs rollouts
    the file structure should be as follows:
        flights/
          ..freq1
              ..rand
              ..roll1
              ..roll2
          ..freq2
              ..rand
              ..roll1
              ..roll2
          ....
    """

    print_flag = False

    if print_flag:
        print('~~~~~~~~~~~~')

    # gather dirs (each will be one line on the plot)
    dirs = [dI for dI in os.listdir(
        csv_dir) if os.path.isdir(os.path.join(csv_dir, dI))]
    # print(dirs)

    font = {'size': 22}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    with sns.axes_style("whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(111)
        plt.subplots_adjust(wspace=.15, left=.1, right=1-.07)  # , hspace=.15)

    colors = ['#208080', '#F83030', '#808000']
    markers = ['*', 'x', '.']

    for i, (dir, c) in enumerate(zip(reversed(sorted(dirs)), colors)):
        if print_flag:
            print('---' + dir + '---')
        # Load each frequency fo rollouts individually
        label = dir + " Rollouts"
        files = os.listdir(csv_dir+dir)

        # create list for dataframes
        data = []
        df_main = pd.DataFrame(columns=["Flight Idx", "Flight Time (ms)",
                                        "Trained Points", "Mean Objective", "RMS Pitch Roll", "roll"])

        means = []
        stds = []
        labels = []
        cum_points = 0

        for f in sorted(files):
            print(f)
            if print_flag:
                print("-> Rollout: " + f[-10:-3])
            with open(csv_dir+dir+'/'+f, "rb") as csvfile:
                # laod data
                roll = f[-f[::-1].find('_'):-f[::-1].find('.')-1]
                df = pd.read_csv(csvfile, sep=",")
                df['roll'] = roll
                df_main = df_main.append(df)

                mean_sub = df["Flight Time (ms)"].mean()
                std_sub = df["Flight Time (ms)"].std()
                data_points = np.sum(df["Trained Points"])

                means.append(mean_sub)
                stds.append(std_sub)
                labels.append(data_points)

                # mean = np.mean(new_data[:,1])
                # std = np.std(new_data[:,1])

                if print_flag:
                    new_data = np.loadtxt(csvfile, delimiter=",", skiprows=1)
                    print("   Mean flight length is: ",
                          np.mean(new_data[:, 1]))
                    print("   Std flight length is: ", np.std(new_data[:, 1]))

        means = np.array(means)
        stds = np.array(stds)
        labels = np.array(labels)
        labels = np.cumsum(labels)
        print(labels)

        x = np.arange(0, len(labels))
        ax1.scatter(labels, means/1000, label=label,
                    marker=markers[i], color=c, linewidth='16')

        # ax1.axhline(np.max(means),linestyle='--', label=str("Max" + dir),alpha=.5, color=c)
        ax1.set_xscale("log", nonposx='clip')
        ax1.set_xlim([50, 20000])

        ax1.set_ylabel("Flight Time (s)")
        ax1.set_xlabel("Trained Datapoints")

        ax1.grid(b=True, which='major', color='k',
                 linestyle='-', linewidth=1.2, alpha=.75)
        ax1.grid(b=True, which='minor', color='b',
                 linestyle='--', linewidth=1.2, alpha=.5)
        # ax1.set_title("Flight Time vs Datapoints")

        # Customize the major grid
        # ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        # plt.grid(True, which='majorminor', linestyle=':', linewidth='0.75', color='black')

        # ax1.set_ylim([0,5000])

        # ax1.set_xticks(x)
        # ax1.set_xticklabels(labels, rotation = 75, fontsize = 14)

        ax1.legend()
        ###############

    plt.show()

    print('\n')

def plot_sensor_quality(dir):
    """
    Goes through subfolders of a given directory to see if there is any noticable changes 
      in how the data is logged that may indicated why some flights are so much better

    Takes the mean and variance of the state data through each takeoff. 
    Will return a matrix of dimesnions (n, dx), so 
    - n number of rollouts 
    - dx is the dimension of the state
    """

    print('------------')
    print('RUNNING TEST OF LOGGEST STATE DATA NOISE')
    dirs = os.listdir("_logged_data_autonomous/"+dir)
    # init arrays for the means of each rollout for large scale analysis
    means = np.zeros([len(dirs), 9])
    noises = np.zeros([len(dirs), 9])

    # init array for a long list of all flights
    total_list = np.zeros([len(dirs)*10, 9])

    # dim = 1
    l1 = []
    l2 = []
    l3 = []

    l7 = []
    l8 = []
    l9 = []

    flight_times = []

    for d in sorted(dirs)[:-1]:
        if d != '.DS_Store':
            print('~~~ dir:', d, ' ~~~')
            files = os.listdir("_logged_data_autonomous/"+dir+'/'+d+'/')
            i = 0
            for f in sorted(files):
                if len(f) > 5 and f[-4:] == '.csv':
                    # print("File num: ", i)
                    new_data = np.loadtxt(
                        "_logged_data_autonomous/"+dir+'/'+d+'/'+f, delimiter=",")

                    Objv = new_data[:, 14]
                    move_idx = np.argmax(Objv != -1)

                    Time = new_data[:, 13]
                    flight_len = Time[-1]-Time[move_idx-5]
                    flight_times.append(flight_len)

                    # GRABS DATA DURING AND BEFORE TAKEOFF
                    state_data = new_data[:move_idx-5, :9]
                    means = np.mean(state_data, axis=0)
                    noise = np.std(state_data, axis=0)
                    # print("Means: ", means)
                    # print("Noises: ", noise)

                    l1.append(noise[0])
                    l2.append(noise[1])
                    l3.append(noise[2])

                    l7.append(noise[6])
                    l8.append(noise[7])
                    l9.append(noise[8])
                    i += 1  # not using enumarate becase DS_Store

    plotting_keys = np.loadtxt(
        "_logged_data_autonomous/"+dir+'/'+"data.csv", delimiter='\t')
    print(np.shape(plotting_keys))
    print(np.shape(l1))

    font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    with sns.axes_style("whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(111)
        plt.subplots_adjust(wspace=.15, left=.1, right=1-.07)  # , hspace=.15)

    for i, row in enumerate(plotting_keys):

        # drift fails
        if row[0] == 1:
            lab1 = plt.axvline(i, linestyle='--', color='r',
                               alpha=.3, label='Drift Failures')

        # sensor fails
        elif row[1] == 1:
            lab2 = plt.axvline(i, linestyle='--', color='b',
                               alpha=.3, label='Sensor Failures')

        # replace parts
        elif row[2] == 1:
            lab3 = plt.axvline(i, linestyle='--', color='k',
                               alpha=.3, label='Replace Parts')

    # Lines for frequency cutoffs
    lab50 = plt.axvline(160, linestyle=':', color='k',
                        alpha=.8, label='50Hz Begins')
    lab75 = plt.axvline(290, linestyle='-.', color='k',
                        alpha=.8, label='75Hz Begins')

    # ax1.set_title("Noise on certain variables across flights")
    ax1.set_xlabel("Chronological Flight")
    ax1.set_ylabel("Std. Dev. Data (deg/s^2) - Noise")
    p1 = plt.plot(l1, label='angular_x')
    p2 = plt.plot(l2, label='angular_y')
    p3 = plt.plot(l3, label='angular_z')

    ax1.set_ylim([0, 6])

    # p4 = plt.plot(l7, label='linear_x')
    # p5 = plt.plot(l8, label='linear_y')
    # p6 = plt.plot(l9, label='linear_z')

    lns = p1+p2+p3+[lab1]+[lab2]+[lab3]+[lab50]+[lab75]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fancybox=True, framealpha=1,
               shadow=True, borderpad=1, ncol=3)

    # ax2 = ax1.twinx()
    # ax2.plot(flight_times, linestyle='-', color = 'k', label='Flight Time')
    # ax2.set_ylabel("Flight Time (ms)")
    plt.show()

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
        with sns.axes_style("whitegrid"):
            plt.rcParams["axes.edgecolor"] = "0.15"
            plt.rcParams["axes.linewidth"] = 1.5
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

        with sns.axes_style("whitegrid"):
            plt.rcParams["axes.edgecolor"] = "0.15"
            plt.rcParams["axes.linewidth"] = 1.5
            fig3, axes3 = plt.subplots(nrows=n_row, ncols=num_subplots, sharey=True, sharex=True)
            # ax31, ax32, ax33, ax34, ax35, ax36 = axes3[:,:]

            # plt.suptitle("Voltage Context Effect on Prediction")
            fig3.text(0.475, 0.05, 'Measured Pitch (Degrees)', ha='center')


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

def plot_waterfall(model, df, equil, var, N, T, plt_idx = []):
    """
    The long overdue plot that takes in a point of a dataframe at random. This is useful for assesing the 
      usefullness of the model predictive controller
    """

    # generate actions in the same manner as the MPC computer
    #  1. sample integers betweeen 0 and num_bins
    #  2. multiple by step size (256)
    #  in our case, we will want an output of dimensions (Nx4) - sample and hold N actions
    
    # need to sample actions individually for bins
    actions_bin_1 = np.random.randint(
        int((equil[0]-var)/256), int((equil[0]+var)/256), (N,1))
    actions_bin_2 = np.random.randint(
        int((equil[1]-var)/256), int((equil[1]+var)/256), (N,1))
    actions_bin_3 = np.random.randint(
        int((equil[2]-var)/256), int((equil[2]+var)/256), (N,1))
    actions_bin_4 = np.random.randint(
        int((equil[3]-var)/256), int((equil[3]+var)/256), (N,1))

    # stack them into an array of (Nx4)
    action_bin = np.hstack(
        (actions_bin_1, actions_bin_2, actions_bin_3, actions_bin_4))
    
    actions = action_bin*256

    # get initial x state
    points = np.squeeze(np.where(df['term'].values == 0))       # not last 
    num_pts = len(points)
    x0_idx = np.random.randint(10,len(points)-20)               # not first few points or towards end

    states = model.state_list                                   # gather these for use
    inputs = model.input_list

    x0 = df[states].values[x0_idx]                              # Values
    u0 = df[inputs].values[x0_idx]

    # initialize large array to store the results in
    predictions = np.zeros((N,T+1,len(x0)))
    predictions[:,0,:] = x0

    truth = df[states[:9]].values[x0_idx:x0_idx+T]

    stack = int(len(x0)/9)
    print(states)
    
    # loop to gather all the predictions
    for n, action in enumerate(actions):
        u = u0
        x = x0
        for t in range(T):

            # get action to pass with history
            u = np.concatenate((action, u[:-5], [u[-1]]))

            predictions[n, t+1, :9] = predict_nn_v2(model, x, u)
            # print(predictions[n, t+1, :9])

            # get state with history
            x = np.concatenate((predictions[n, t+1, :9], x[:-9]))
            # print(u)
            # print(u.shape)
            # print(x)
            # print(x.shape)
        # quit()


    # *******************************************************************************************
    # PLOTTING
        # New plot
    font = {'size': 23, 'family': 'serif', 'serif': ['Times']}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=4.5)

    plt.tight_layout()

    # plot for test train compare

    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        plt.subplots_adjust(wspace=.15, left=.1, right=1-.07)  # , hspace=.15)
        ax1 = plt.subplot(111)

    N = np.shape(predictions)[0]
    my_dpi = 96
    plt.figure(figsize=(3200/my_dpi, 4000/my_dpi), dpi=my_dpi)
    dim = 4
    pred_dim = predictions[:, :, dim]
    
    i=0
    for traj in pred_dim:
        if i==0:
            ax1.plot(traj, linestyle=':', linewidth=4,
                     label='Predicted State', alpha=.75)
        else:
            ax1.plot(traj, linestyle=':', linewidth=4, alpha=.75)
        i += 1

    ax1.plot(truth[:,dim], linestyle='-', linewidth=4.5, color='k', marker = 'o', alpha=.8, markersize='10',label = 'Ground Truth')
    ax1.set_ylim([-40,40])

    # find best action
    print(predictions[:, 3:5]**2)
    print(np.sum(np.sum(predictions[:,:, 3:5]**2,axis=2),axis=1))
    best_id = np.argmin(np.sum(np.sum(predictions[:, :, 3:5]**2, axis=2), axis=1))
    ax1.plot(predictions[best_id, :, dim], linestyle='-', linewidth=4.5, color='r', alpha = .8, label='Chosen Action')
    ax1.legend()
    ax1.set_ylabel('Roll (deg)')
    ax1.set_xlabel('Timestep (T)')
    # ax1.set_xticks(np.arange(0, 5.1, 1))
    # ax1.set_xticklabels(["s(t)", "1", "2", "3", "4", "5"])
    

    ax1.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.75)
    ax1.grid(b=True, which='minor', color='b',
             linestyle='--', linewidth=0, alpha=.5)
    ax1.set_xlim([0,20])

    plt.show()

def plot_traj_model(df_traj, model):
    """plots all the states predictions over time"""

    state_list, input_list, target_list = model.get_training_lists()
    data_params = {
        'states': state_list,
        'inputs': input_list,
        'targets': target_list,
        'battery': True
    }

    X, U, dX = df_to_training(df_traj, data_params)

    num_skip = 0
    X, U, dX = X[num_skip:, :], U[num_skip:, :], dX[num_skip:, :]
    # Gets starting state
    x0 = X[0, :]

    # get dims
    stack = int((len(X[0, :]))/9)
    xdim = 9
    udim = 4

    # store values
    pts = len(df_traj)-num_skip
    x_stored = np.zeros((pts, stack*xdim))
    x_stored[0, :] = x0
    x_shift = np.zeros(len(x0))

    ####################### Generate Data #######################
    for t in range(pts-1):
        # predict
        # x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], U[t,:])
        x_pred = predict_nn_v2(model, x_stored[t, :], U[t, :])

        if stack > 1:
            # shift values
            x_shift[:9] = x_pred
            x_shift[9:-1] = x_stored[t, :-10]
        else:
            x_shift = x_pred

        # store values
        x_stored[t+1, :] = x_shift

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

    ax1.set_ylim([-150, 150])
    ax2.set_ylim([-150, 150])
    ax3.set_ylim([-150, 150])
    ax4.set_ylim([-35, 35])
    ax5.set_ylim([-35, 35])
    # ax6.set_ylim([-35,35])
    ax7.set_ylim([-6, 6])
    ax8.set_ylim([-6, 6])
    ax9.set_ylim([5, 15])

    ax1.plot(x_stored[:, 0], linestyle='--', color='b', label='Predicted')
    ax1.plot(X[:, 0], color='k', label='Ground Truth')

    ax2.plot(x_stored[:, 1], linestyle='--', color='b', label='Predicted')
    ax2.plot(X[:, 1], color='k', label='Ground Truth')

    ax3.plot(x_stored[:, 2], linestyle='--', color='b', label='Predicted')
    ax3.plot(X[:, 2], color='k', label='Ground Truth')

    ax4.plot(x_stored[:, 3], linestyle='--', color='b', label='Predicted')
    ax4.plot(X[:, 3], color='k', label='Ground Truth')

    ax5.plot(x_stored[:, 4], linestyle='--', color='b', label='Predicted')
    ax5.plot(X[:, 4], color='k', label='Ground Truth')

    ax6.plot(x_stored[:, 5], linestyle='--', color='b', label='Predicted')
    ax6.plot(X[:, 5], color='k', label='Ground Truth')

    ax7.plot(x_stored[:, 6], linestyle='--', color='b', label='Predicted')
    ax7.plot(X[:, 6], color='k', label='Ground Truth')

    ax8.plot(x_stored[:, 7], linestyle='--', color='b', label='Predicted')
    ax8.plot(X[:, 7], color='k', label='Ground Truth')

    ax9.plot(x_stored[:, 8], linestyle='--', color='b', label='Predicted')
    ax9.plot(X[:, 8], color='k', label='Ground Truth')

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
        raise ValueError(
            "Did not include battery voltage for battery plotting")

    X, U, dX = df_to_training(df_traj, data_params)

    # plot properties
    font = {'size': 22}

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

    thrust = np.mean(U[:, :4], axis=1)
    vbat = U[:, -1]

    ####################### PLOT #######################
    with sns.axes_style("whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(111)
        # ax2 = plt.subplot(212)

    # plt.title("Comparing Battery Voltage to Thrust")

    # ax1.set_ylim([-150, 150])
    # ax2.set_ylim([-150, 150])
    time = np.linspace(0, len(thrust)*.02, len(thrust))
    ln1 = ax1.plot(time, thrust, color='r', label='Crazyflie Thrust',
                   markevery=3, marker='*', markersize='20')
    ax1.set_ylabel("Crazyflie Thrust (PWM)", color='k')
    ax1.tick_params('y', colors='r')

    ax1.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=1, alpha=.5)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        time, vbat, label='Crazyflie Battery Voltage', color='b', markevery=3, marker='.', markersize='20')
    ax2.set_ylabel("Crazyflie Battery Voltage (mV)", color='k')
    ax2.tick_params('y', colors='b')

    ax1.set_xlabel("Time (s)")

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)  # , loc=5)

    plt.show()


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


def plot_test_train(model, dataset):
    """
    Takes a dynamics model and plots test vs train predictions on a dataset of the form (X,U,dX)
    """
    '''
    Some Models:
    model_pll = '_models/temp/2018-12-14--10-47-41.7_plot_pll_stack3_.pth'
    model_mse = '_models/temp/2018-12-14--10-51-10.9_plot_mse_stack3_.pth'
    model_pll_ens = '_models/temp/2018-12-14--10-53-42.9_plot_pll_ensemble_stack3_.pth'
    model_pll_ens_10 = '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth'
    model_mse_ens = '_models/temp/2018-12-14--10-52-40.4_plot_mse_ensemble_stack3_.pth'
    '''
    model_pll_ens_10 = '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth'

    predictions_pll_ens = gather_predictions(model_pll_ens_10, dataset)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

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
    font = {'size': 22, 'family': 'serif', 'serif': ['Times']}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=4.5)

    plt.tight_layout()

    # plot for test train compare

    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure()
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        plt.subplots_adjust(left=.1, right=1-.07, hspace=.28)

    gt_train = np.linspace(0, 1, len(gt_sort_train))
    ax1.plot(gt_train,gt_sort_train, label='Ground Truth', color='k', linewidth=3.8)
    ax1.plot(gt_train, pred_sort_pll_train, '-', label='Probablistic Model Prediction',
             markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
    ax1.set_title("Training Data Predictions")
    ax1.legend()

    gt_test = np.linspace(0, 1, len(gt_sort_test))
    ax2.plot(gt_test, gt_sort_test, label='Ground Truth', color='k', linewidth=3.8)
    ax2.plot(gt_test, pred_sort_pll_test, '-', label='Bayesian Model Validation DataPrediction',
             markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
    ax2.set_title("Test Data Predictions")

    fontProperties = {'family': 'Times New Roman'}

    # a = plt.gca()
    # print(a)
    # a.set_xticklabels(a.get_xticks(), fontProperties)
    # a.set_yticklabels(a.get_yticks(), fontProperties)

    ax1.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.75)
    ax1.grid(b=True, which='minor', color='b',
             linestyle='--', linewidth=0, alpha=.5)

    ax2.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.75)
    ax2.grid(b=True, which='minor', color='b',
             linestyle='--', linewidth=0, alpha=.5)

    fig.text(.02, .7, 'One Step Prediction, Pitch (deg)',
             rotation=90, family='Times New Roman')
    fig.text(.404, .04, 'Sorted Datapoints, Normalized', family='Times New Roman')

    for ax in [ax1, ax2]:
        # if ax == ax1:
        #     loc = matplotlib.ticker.MultipleLocator(base=int(lx/10))
        # else:
        #     loc = matplotlib.ticker.MultipleLocator(
        #         base=int((np.shape(dX)[0]-lx)/10))
        ax.set_ylim([-6.0, 6.0])
        ax.set_xlim([0, 1])

    plt.show()


def plot_rollout_compare():
    """
    Function to plot the first 4 rollout flights to show rapid learning. 
      It assumes a specific file structure so will run on it's own.
      Includes some commented code for an old version.
    """

    font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=3)

    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

    axes = [ax1, ax2, ax3, ax4]
    plt.tight_layout()

    line1, = ax1.plot([], [])

    dim = 3

    # original plot
    dir1 = "_logged_data_autonomous/_examples/icra-top20/roll0/"
    dir2 = "_logged_data_autonomous/_examples/icra-top20/roll1/"
    dir3 = "_logged_data_autonomous/_examples/icra-top20/roll2/"
    dir4 = "_logged_data_autonomous/_examples/icra-top20/roll3/"
    dir5 = "_logged_data_autonomous/_examples/icra-top20/roll4/"
    dir6 = "_logged_data_autonomous/_examples/icra-top20/roll5/"

    #new plot
    dir1 = "_logged_data_autonomous/_newquad1/publ2/c50_rand/"
    dir2 = "_logged_data_autonomous/_newquad1/publ2/c50_roll01/"
    dir3 = "_logged_data_autonomous/_newquad1/publ2/c50_roll02/"
    dir4 = "_logged_data_autonomous/_newquad1/publ2/c50_roll03/"
    dir5 = "_logged_data_autonomous/_newquad1/publ2/c50_roll04/"
    # dir6 = "_logged_data_autonomous/_newquad1/publ2/c50_roll05/"
    # dir7 = "_logged_data_autonomous/_newquad1/publ2/c50_roll06/"

    dirs = [dir1, dir2, dir3, dir4]  # , dir5]#, dir6]#, dir7]
    colors = ['r', 'y', 'g', 'c']  # , 'b']#, 'm']#, 'k' ]
    colors = ['r', 'b', 'g', 'k']  # , 'b']#, 'm']#, 'k' ]
    best_len = 0
    best_time = 3000

    load_params = {
        'delta_state': True,                # normally leave as True, prediction mode
        # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
        'include_tplus1': True,
        # trims high vbat because these points the quad is not moving
        'trim_high_vbat': 4200,
        # If not trimming data with fast log, need another way to get rid of repeated 0s
        'takeoff_points': 180,
        # if all the euler angles (floats) don't change, it is not realistic data
        'trim_0_dX': True,
        'find_move': True,
        # if the states change by a large amount, not realistic
        'trime_large_dX': True,
        # Anything out of here is erroneous anyways. Can be used to focus training
        'bound_inputs': [25000, 65500],
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

    # load_params ={
    #     'delta_state': True,
    #     'takeoff_points': 0,
    #     'trim_0_dX': True,
    #     'trime_large_dX': True,
    #     'bound_inputs': [20000,65500],
    #     'stack_states': 4,
    #     'collision_flag': False,
    #     'shuffle_here': False,
    #     'timestep_flags': [],
    #     'battery' : False
    # }

    for k, dir in enumerate(dirs):
        axis = axes[k]
        for i in range(10):
            # file = random.choice(os.listdir(dir))
            file = os.listdir(dir)[i]
            print(file)
            print('Processing File: ', file, 'Dir: ', k, 'File number: ', i)
            if dir == dir4 or dir == dir5 or dir == dir6:
                 takeoff = True
                 load_params['takeoff_points'] = 170
            else:
                 takeoff = False

            X, U, dX, objv, Ts, times, terminal = trim_load_param(
                str(dir+file), load_params)

            time = np.max(times)
            n = len(X[:, dim])
            if n > best_len:
                best_len = n
            if time > best_time:
                best_time = time
                print(best_time)
            x = np.linspace(0, time, len(Ts))
            if i == 0:
                axis.plot(x/1000, X[:, dim], linestyle='-', alpha=.5,
                          linewidth=3, color=colors[k], label="Rollout %d" % k)
                # axis.plot(x, X[:, dim], color='k', linestyle='-')
            else:
                axis.plot(x/1000, X[:, dim], linestyle='-', alpha=.5,
                          linewidth=3, color=colors[k])
                # axis.plot(x, X[:, dim], color='k', linestyle='-')

    # print(x)
    # scaled_time = np.round(np.arange(0, best_time, best_time/best_len),1)
    # print(scaled_time)
    # ax1.set_xticks(scaled_time[0::10])
    # ax1.set_xticklabels([str(x) for x in scaled_time[::10]])
    # leg = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol = 6)#len(dirs))
    # # get the individual lines inside legend and set line width
    # for line in leg.get_lines():
    #     line.set_linewidth(2.5)

    # ax1.set_xlabel("Time (ms)")
    # ax1.set_xlim([0,best_time])
    # ax1.set_xlim([0,2000])
    # ax1.set_ylim([-40,40])
    # ax1.set_ylabel("Pitch (Deg)")

    #Version two of the figure
    for i, ax in enumerate(axes):
        if i > 1:
            ax.set_xlabel("Time (s)")
        ax.set_xlim([0, best_time])
        ax.set_xlim([0, 2.500])
        ax.set_ylim([-40, 40])
        if i == 0 or i == 2:
            ax.set_ylabel("Pitch (Deg)")

        ax.grid(b=True, which='major', color='k',
                linestyle='-', linewidth=0.2, alpha=.5)
        ax.grid(b=True, which='minor', color='r', linestyle='--', linewidth=0)

    plt.subplots_adjust(wspace=.15, left=.07, right=1-.07)  # , hspace=.15)
    ax1.set_title("Random Controller Flights")
    ax2.set_title("After 1 Model Iteration")
    ax3.set_title("After 2 Model Iterations")
    ax4.set_title("After 3 Model Iterations")
    # plt.suptitle("Comparision of Flight Lengths in Early Rollouts")

    # fig.set_size_inches(15, 7)

    # plt.savefig('psoter', edgecolor='black', dpi=100, transparent=True)

    plt.show()


def plot_flight_segment(fname, load_params):
    """
    Takes in a file and loading instructions and plots a segment of the flight PWMs and Euler Angles
    """
    # load_params = {
    #     'delta_state': True,                # normally leave as True, prediction mode
    #     # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    #     'include_tplus1': True,
    #     # trims high vbat because these points the quad is not moving
    #     'trim_high_vbat': 4400,
    #     # If not trimming data with fast log, need another way to get rid of repeated 0s
    #     'takeoff_points': 180,
    #     # if all the euler angles (floats) don't change, it is not realistic data
    #     'trim_0_dX': True,
    #     'find_move': True,
    #     # if the states change by a large amount, not realistic
    #     'trime_large_dX': True,
    #     # Anything out of here is erroneous anyways. Can be used to focus training
    #     'bound_inputs': [15000, 65500],
    #     # IMPORTANT ONE: stacks the past states and inputs to pass into network
    #     'stack_states': 3,
    #     # looks for sharp changes to tthrow out items post collision
    #     'collision_flag': False,
    #     # shuffle pre training, makes it hard to plot trajectories
    #     'shuffle_here': False,
    #     'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    #     'battery': True,                   # if battery voltage is in the state data
    #     # adds a column to the dataframe tracking end of trajectories
    #     'terminals': True,
    #     'fastLog': False,                   # if using the software with the new fast log
    #     # Number of times the control freq you will be using is faster than that at data logging
    #     'contFreq': 1
    # }

    # original for ICRA
    # fname = '_logged_data_autonomous/sep14_150_3/flight_log-20180914-182941.csv'

    # file for arxiv draftt
    # fname = '_logged_data_autonomous/_newquad1/publ2/c50_roll06/flight_log-20181115-101931.csv'


    X, U, dX, objv, Ts, time, terminal = trim_load_param(fname, load_params)


    font = {'size': 22,'family': 'serif', 'serif': ['Times']}
    
    
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=5)
    plt.tight_layout()

    # with sns.axes_style("whitegrid"):
    #     plt.rcParams["font.family"] = "Times New Roman"
    #     plt.rcParams["axes.edgecolor"] = "0.15"
    #     plt.rcParams["axes.linewidth"] = 1.5
    #     fig = plt.figure()
    #     ax1 = plt.subplot(211)
    #     ax2 = plt.subplot(212)


    # for video figure
    my_dpi = 200
    fig = plt.figure(figsize=(3.5*1920/my_dpi, 2*560/my_dpi), dpi=my_dpi)
    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax2 = plt.subplot(111)

    print(np.max(time))

    n = len(X[:, 3])
    scaled_time = np.linspace(0, n, n)*20/5/1000
    # ax1.set_title('Example Flight Performance')
    # plt.title('Autonomous Flight Data')

    # SHORTER
    shorter = int(n/5)
    # ax1.plot(scaled_time, U[:,0], label= 'm1', alpha =.8)
    # ax1.plot(scaled_time, U[:,1], label= 'm2', alpha =.8)
    # ax1.plot(scaled_time, U[:,2], label= 'm3', alpha =.8)
    # ax1.plot(scaled_time, U[:,3], label= 'm4', alpha =.8)

    # SHORT
    # ax1.plot(scaled_time[:shorter], U[:shorter, 0], label='m1',
    #          alpha=.8, markevery=20, marker='.', markersize='20')
    # ax1.plot(scaled_time[:shorter], U[:shorter, 1], label='m2',
    #          alpha=.8, markevery=20, marker='*', markersize='20')
    # ax1.plot(scaled_time[:shorter], U[:shorter, 2], label='m3',
    #          alpha=.8, markevery=20,  marker='^', markersize='20')
    # ax1.plot(scaled_time[:shorter], U[:shorter, 3], label='m4',
    #          alpha=.8, markevery=20, marker='1', markersize='20')
    # ax1.set_ylim([20000, 57000])

    # ax1.set_ylabel('Motor Power (PWM)')

    ax2.set_ylim([-30, 30])
    ax2.set_ylim([-25, 25])
    ax2.set_ylabel('Euler Angles (Deg)')
    ax2.set_xlabel('Time (s)')
    # fig.text(.44, .03, 'Time (s)', family="Times New Roman")

    # LONG
    ax2.plot(scaled_time, X[:, 3], label='Pitch', marker='.', markevery = 25, markersize='20')
    ax2.plot(scaled_time, X[:, 4], label='Roll',
             marker='^', markevery= 25,  markersize='20')

    # SHORT
    # ax2.plot(scaled_time[:shorter], X[:shorter, 3],
    #          label='Pitch', markevery=20, marker='.', markersize='20')
    # ax2.plot(scaled_time[:shorter], X[:shorter, 4],
    #          label='Roll', markevery=20,  marker='^', markersize='20')
    # YAWwWww ax2.plot(scaled_time, X[:,5]-X[0,5])


    # leg1 = ax1.legend(ncol=4, loc=0)
    leg2 = ax2.legend(loc=8, ncol=2, frameon=True)  # , 'Yaw'])

    ax2.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.5)

    # ax1.grid(b=True, which='major', color='k',
    #          linestyle='-', linewidth=0, alpha=.5)

    # ax1.set_xlim([0, 1])
    # ax2.set_xlim([0, 1])

    ax2.set_xlim([-0.4,6])
    # for line in leg1.get_lines():
    #     line.set_linewidth(2.5)

    # for line in leg2.get_lines():
    #     line.set_linewidth(2.5)
    # ax1.grid(True, ls= 'dashed')
    # # ax2.grid(True, ls= 'dashed')
    # ax3.grid(True, ls= 'dashed')
    # ax4.grid(True, ls= 'dashed')

    # plt.subplots_adjust(wspace=.15, left=.07, right=1-.07)  # , hspace=.15)

    plt.savefig('_results/poster', edgecolor='black', dpi=my_dpi, transparent=False)

    plt.show()
