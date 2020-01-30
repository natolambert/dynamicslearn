
import utils.matplotlib as u_p
import utils.data as u_d
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import os
import sys

# sys.path.append(os.getcwd())


# load initial state or generate.
load_params = {
    'delta_state': True,                # normally leave as True, prediction mode
    'include_tplus1': True,
    'find_move': True,
    # If not trimming data with fast log, need another way to get rid of repeated 0s
    'takeoff_points': 180,
    # trims high vbat because these points the quad is not moving
    'trim_high_vbat': 4000,
    # if all the euler angles (floats) don't change, it is not realistic data
    'trim_0_dX': True,
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
    'contFreq': 1,
    'zero_yaw': False
}


## PLOT for cross verification of moving average filtering of euler angles

# # these numbers were filtering only the ACCELERATIONS
# ens_avgs = np.array([-29.929224, -30.69624, -30.291714, -29.814226, -30.231466, -30.337006])

# s_avgs = np.array([-33.870785, -35.147144, -33.032555, -36.463993, -33.438164, -35.17601])

# ens_avgs = ens_avgs - np.mean(ens_avgs)
# s_avgs = s_avgs - np.mean(s_avgs)
# # font = {'size': 22,'family': 'serif', 'serif': ['Times']}
# font = {'size': 12}

# matplotlib.rc('font', **font)
# matplotlib.rc('lines', linewidth=1.25)
# matplotlib.rc('text', usetex=True)

# # plt.tight_layout()

# fig = plt.figure()
# with sns.axes_style("whitegrid"):
#     plt.rcParams["font.family"] = "Times New Roman"
#     plt.rcParams["axes.edgecolor"] = "0.15"
#     plt.rcParams["axes.linewidth"] = 1.5
#     plt.subplots_adjust(top=.93, bottom=0.13, left=.13,
#                         right=1-.03, hspace=.25)
#     ax1 = plt.subplot(111)

# ax1.set_title("Likelihood of Test Sets")
# ax1.set_xlabel("Moving Average on Accelerations")
# ax1.set_ylabel("Log Likelihood Relative to Mean - Lower Better")
# # ax1.set_ylim([-29,-38])
# ax1.plot(ens_avgs, label = 'Ensemble, E=7')
# ax1.plot(s_avgs, label='Single Nets')
# ax1.set_xticks([0,1,2,3,4,5])
# ax1.set_xticklabels(["","2", "3", "4", "5", "6"], rotation=0, fontsize=12)
# ax1.legend()
# plt.show()

# quit()

# Ionocraft model learning:
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
    'trime_large_dX': False,
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
    'contFreq': 1,
    'iono_data': True,
    'zero_yaw': True,
    'moving_avg': 2
}
# model = "_models/temp/2019-05-02--09-43-01.5_temp_stack3_.pth"
# df = u_d.stack_dir_pd_iono('video-setup/', load_params)

#
# nn = torch.load(
#     model)
# nn.eval()
# state_list, input_list, change_list = nn.get_training_lists()
#
# data_params = {'states': state_list, 'inputs': input_list,
#                'targets': change_list, 'battery': False}
#
#
# (X, U, dX) = u_d.df_to_training(df, data_params)
# # print(X)
#
#
# u_p.plot_test_train(nn, (X, U, dX), variances=False)
# quit()

# dir_list = ["_newquad1/fixed_samp/c100_samp300_rand/",
#             "_newquad1/fixed_samp/c100_samp300_roll1/", 
#             "_newquad1/fixed_samp/c100_samp300_roll2/"]

# dir_list = ["_newquad1/publ_data/c50_samp300_rand/",
#             "_newquad1/publ_data/c50_samp300_roll1/",
#             "_newquad1/publ_data/c50_samp300_roll2/",
#             "_newquad1/publ_data/c50_samp300_roll3/",
#             "_newquad1/publ_data/c50_samp300_roll4/"]

# dir_list = ["_newquad1/publ_data/c25_samp300_rand/",
#     "_newquad1/publ_data/c25_samp300_roll1/",
#     "_newquad1/publ_data/c25_samp300_roll2/",
#     "_newquad1/publ_data/c25_samp300_roll3/",
#     "_newquad1/publ_data/c25_samp300_roll4/"]

# dir_list = ["_newquad1/publ2/c50_rand/",  # ,
#             "_newquad1/publ2/c50_roll01/",
#             "_newquad1/publ2/c50_roll02/",
#             "_newquad1/publ2/c50_roll03/",
#             "_newquad1/publ2/c50_roll04/",
#             "_newquad1/publ2/c50_roll05/",
#             "_newquad1/publ2/c50_roll06/",
#             "_newquad1/publ2/c50_roll07/",
#             "_newquad1/publ2/c50_roll08/",
#             "_newquad1/publ2/c50_roll09/",
#             "_newquad1/publ2/c50_roll10/",
#             "_newquad1/publ2/c50_roll11/",
#             "_newquad1/publ2/c50_roll12/"]

dir_list = ["_newquad1/publ2/c25_rand/",
            "_newquad1/publ2/c25_roll01/",
            "_newquad1/publ2/c25_roll02/",
            "_newquad1/publ2/c25_roll03/",
            "_newquad1/publ2/c25_roll04/",
            "_newquad1/publ2/c25_roll05/",
            "_newquad1/publ2/c25_roll06/",
            "_newquad1/publ2/c25_roll07/",
            "_newquad1/publ2/c25_roll08/",
            "_newquad1/publ2/c25_roll09/",
            "_newquad1/publ2/c25_roll10/",
            "_newquad1/publ2/c25_roll11/",
            "_newquad1/publ2/c25_roll12/"]

# dir_list = ["_newquad1/publ2/c50_roll09/",
#             "_newquad1/publ2/c50_roll10/"]


# other_dirs = ["150Hz/sep13_150_2/","/150Hzsep14_150_2/","150Hz/sep14_150_3/"]
df = u_d.load_dirs(dir_list, load_params)

nn = torch.load(
    '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth')
nn.eval()

state_list, input_list, change_list = nn.get_training_lists()


data_params = {'states': state_list, 'inputs': input_list,
               'targets': change_list, 'battery': False}

X, U, dX = u_d.df_to_training(df, data_params)
###### PREDICTIONS TEST TRAIN ##########

u_p.plot_test_train(nn, (X,U,dX), variances = True)


###### WATERFALL ##########

df_traj,_ = u_d.get_rand_traj(df)
# X, U, dX = u_d.df_to_training(df_traj, data_params)

PWMequil = np.array([34687.1, 37954.7, 38384.8, 36220.11])  # new quad
# nn = torch.load('_models/temp/2018-12-30--10-02-51.1_true_plot_50_stack3_.pth')
# nn.eval()

# plot_traj_model(df_traj, nn)
# plot_battery_thrust(df_traj, nn)
u_p.plot_waterfall(nn, df_traj, PWMequil, 5000, 50, 20, plt_idx=[])

####### LEARNING PLOTS ##########
# u_p.plot_flight_time("_results/_summaries/trainedpoints/")
# trained_points_plot("_results/_summaries/trainedpoints/")
quit()
####### ROLLOUT PLOT ##########
# u_p.plot_rollout_compare()

####### FLIGHT PLOT ##########
load_params = {
    'delta_state': True,                # normally leave as True, prediction mode
    # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'include_tplus1': True,
    # trims high vbat because these points the quad is not moving
    'trim_high_vbat': 4400,
    # If not trimming data with fast log, need another way to get rid of repeated 0s
    'takeoff_points': 180,
    # if all the euler angles (floats) don't change, it is not realistic data
    'trim_0_dX': True,
    'find_move': True,
    # if the states change by a large amount, not realistic
    'trime_large_dX': True,
    # Anything out of here is erroneous anyways. Can be used to focus training
    'bound_inputs': [15000, 65500],
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
    'fastLog': False,                   # if using the software with the new fast log
    # Number of times the control freq you will be using is faster than that at data logging
    'contFreq': 1
}

# file for short arxiv draftt
fname = '_logged_data_autonomous/_newquad1/publ2/c50_roll06/flight_log-20181115-101931.csv'
u_p.plot_flight_segment(fname, load_params)

# File for long plot arxiv draft
fname = '_logged_data_autonomous/_newquad1/fixed_samp/c100_samp275_misc/flight_log-20181031-132327.csv'
# u_p.plot_flight_segment(fname, load_params)
