
# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datetime import timedelta
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import csv
from scipy.signal import butter, lfilter, freqz


def stack_dir_pd(dir, load_params):
    '''
    Takes in a directory and returns a dataframe for the data
    '''
    print('Loading dir: ', dir)
    files = os.listdir("_logged_data_autonomous/"+dir)
    print('...number of flights: ', len(files))

    # init arrays
    X = []
    U = []
    dX = []
    objv = []
    Ts = []
    times = []
    terminals = []

    # init if needed. This will play with terminals a little bit
    if load_params['include_tplus1']:
        tplus1 = []

    for f in files:
        # print(f)
        if len(f) > 5 and f[-4:] == '.csv':
            X_t, U_t, dX_t, objv_t, Ts_t, time, terminal = trim_load_param("_logged_data_autonomous/"+dir+f, load_params)

            # shortens length by one point
            if load_params['include_tplus1']:
                if times == []:
                    tplus1 = X_t[1:,:]
                else:
                    tplus1 = np.append(tplus1, X_t[1:,:],axis=0)

                X_t = X_t[:-1,:]
                U_t = U_t[:-1,:]
                dX_t = dX_t[:-1,:]
                objv_t = objv_t[:-1]
                Ts_t = Ts_t[:-1]
                time = time[:-1]
                terminal = terminal[:-1]
                terminal[-1] = 1

            # global time (ROS time)
            if times == []:
                times = time
            else:
                times = np.append(times,time)

            # State data
            if X == []:
                X = X_t
            else:
                X = np.append(X, X_t, axis=0)

            # inputs
            if U == []:
                U = U_t
            else:
                U = np.append(U, U_t, axis=0)

            # change in state
            if dX == []:
                dX = dX_t
            else:
                dX = np.append(dX, dX_t, axis=0)

            # time step
            if Ts_t == []:
                Ts = Ts_t
            else:
                Ts = np.append(Ts, Ts_t, axis=0)

            # objective value
            if objv_t == []:
                objv = objv_t
            else:
                objv = np.append(objv, objv_t, axis=0)

            # end of trajectory marker
            if terminals == []:
                terminals = terminal
            else:
                terminals = np.append(terminals, terminal, axis=0)

    print('...has additional trimmed datapoints: ', np.shape(X)[0])

    ######################################################################

    if False:
        font = {'size'   : 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', linewidth=2.5)

        # plt.tight_layout()

        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot(111)

        ax1.hist(Ts, bins=250)
        ax1.set_title("Hist of time between log packets")
        ax1.set_xlabel("Time Difference Between Packets (ms)")
        ax1.set_ylabel("Occurence")
        plt.show()

    # Start dataframe

    stack_states = load_params['stack_states']
    if stack_states > 0:
        state_idxs = np.arange(0,9*stack_states, 9)
        input_idxs = np.arange(0,4*stack_states, 4)

        d = {'d_omega_x': dX[:,0],
            'd_omega_y': dX[:,1],
            'd_omega_z': dX[:,2],
            'd_pitch': dX[:,3],
            'd_roll': dX[:,4],
            'd_yaw': dX[:,5],
            'd_lina_x': dX[:,6],
            'd_lina_y': dX[:,7],
            'd_liny_z': dX[:,8],

            'timesteps': Ts[:],
            'objective vals': objv[:],
            'flight times': times[:]
            }

        k = 0
        for i in state_idxs:
            st = str(k)
            k+=1
            d['omega_x'+st] = X[:,0+i]
            d['omega_y'+st] = X[:,1+i]
            d['omega_z'+st] = X[:,2+i]
            d['pitch'+st] = X[:,3+i]
            d['roll'+st] = X[:,4+i]
            d['yaw'+st] = X[:,5+i]
            d['lina_x'+st] = X[:,6+i]
            d['lina_y'+st] = X[:,7+i]
            d['lina_z'+st] = X[:,8+i]

        k = 0
        for j in input_idxs:
            st = str(k)
            k+=1
            d['m1_pwm_'+st] = U[:,0+j]
            d['m2_pwm_'+st] = U[:,1+j]
            d['m3_pwm_'+st] = U[:,2+j]
            d['m4_pwm_'+st] = U[:,3+j]

    else: #standard
        d = {'omega_x': X[:,0],
            'omega_y': X[:,1],
            'omega_z': X[:,2],
            'pitch': X[:,3],
            'roll': X[:,4],
            'yaw': X[:,5],
            'lina_x': X[:,6],
            'lina_y': X[:,7],
            'liny_z': X[:,8],

            'm1_pwm': U[:,0],
            'm2_pwm': U[:,1],
            'm3_pwm': U[:,2],
            'm4_pwm': U[:,3],

            'd_omega_x': dX[:,0],
            'd_omega_y': dX[:,1],
            'd_omega_z': dX[:,2],
            'd_pitch': dX[:,3],
            'd_roll': dX[:,4],
            'd_yaw': dX[:,5],
            'd_lina_x': dX[:,6],
            'd_lina_y': dX[:,7],
            'd_lina_z': dX[:,8],

            'timesteps': Ts[:],
            'objective vals': objv[:],
            'flight times': times[:]
            }

    # if including tplus 1 (for predicting some raw next states rather than change)
    if load_params['include_tplus1']:
        d['t1_omega_x'] = tplus1[:,0]
        d['t1_omega_y'] = tplus1[:,1]
        d['t1_omega_z'] = tplus1[:,2]
        d['t1_pitch'] = tplus1[:,3]
        d['t1_roll'] = tplus1[:,4]
        d['t1_yaw'] = tplus1[:,5]
        d['t1_lina_x'] = tplus1[:,6]
        d['t1_lina_y'] = tplus1[:,7]
        d['t1_lina_z'] = tplus1[:,8]

    # terminals is useful for training and testing trajectories
    track_terminals = load_params['terminals']
    if track_terminals: d['term'] = terminals

    # loads battery if needed
    battery = load_params['battery']
    if battery:
        d['vbat'] = X[:,-1]



    df = pd.DataFrame(data=d)
    # print(df)
    print('\n')
    return df

def trim_load_param(fname, load_params):
    '''
    Opens the directed csv file and returns the arrays we want

    Returns: X_t, U_t, dX_t, objv_t, Ts_t, time, terminal
    '''

    # Grab params
    delta_state = load_params['delta_state']
    include_tplus1 = load_params['include_tplus1']
    takeoff_points = load_params['takeoff_points']
    trim_0_dX = load_params['trim_0_dX']
    find_move = load_params['find_move']
    trime_large_dX = load_params['trime_large_dX']
    bound_inputs = load_params['bound_inputs']
    input_stack = load_params['stack_states']
    collision_flag = load_params['collision_flag']
    shuffle_here = load_params['shuffle_here']
    timestep_flags = load_params['timestep_flags']
    battery = load_params['battery']
    fastLog = load_params['fastLog']
    contFreq = load_params['contFreq']
    bat_trim = load_params['trim_high_vbat']
    zero_yaw = load_params['zero_yaw']

    with open(fname, "rb") as csvfile:
        # laod data
        new_data = np.loadtxt(csvfile, delimiter=",")

        # zero yaw to starting condition
        if zero_yaw:
            new_data[:,5] = new_data[:,5] - new_data[0,5]
            # raise NotImplementedError("Need to implement Yaw zeroing with wrap around of angles")

        ########### THESE BARS SEPARATE TRIMMING ACTIONS #########################
        # For now, remove the last 4 columns becausee they're PWMS
        if np.shape(new_data)[1] == 20:
            new_data = new_data[:,:16]

        if bat_trim > 0:
            vbat = new_data[:,-1]
            new_data = new_data[vbat<bat_trim,:]

        # add pwm latency calculations
        pwm_rec = new_data[:,9:13]
        pwm_com = new_data[:,16:]
        # for each pwm in pwm_com
        # find the earliest next index in the pwm_rec
        # for each command record the delta index in a new array
        #    this new array should be of length Uchange?

        # Finds the points where the input changes
        if fastLog:
            Uchange = np.where(new_data[:-1,9:13] != new_data[1:,9:13])
            Uchange = np.unique(Uchange)
            # print(np.shape(Uchange))
            # print(Uchange)

            # If control freq is faster, sample twice in the interval for each unique PWM point
            if contFreq > 1:
                if contFreq == 2:       # training for twice control rate
                    dT = Uchange[1:]-Uchange[:-1]
                    add = Uchange[1:] - np.round(dT/2)
                    Uchange = np.concatenate([Uchange, add])
                    Uchange = np.sort(Uchange).astype(int)
                    new_data = new_data[Uchange, :]

                if contFreq == 3:       # training for three times control rate (150Hz when sampled at 50)
                    dT = Uchange[1:]-Uchange[:-1]
                    add = Uchange[1:] - np.round(dT/3)
                    add2 = Uchange[1:] - np.round(2*dT/3)
                    Uchange = np.concatenate([Uchange, add, add2])
                    Uchange = np.sort(Uchange).astype(int)
                    new_data = new_data[Uchange, :]

            # Else sample each unique point once
            else:
                new_data = new_data[Uchange, :]


        ###########################################################################
        # adding to make the input horizontally stacked set of inputs, rather than only the last input because of spinup time
        if input_stack >1:
            n, du = np.shape(new_data[:,9:13])
            _, dx = np.shape(new_data[:,:9])
            U = np.zeros((n-input_stack+1,du*input_stack))
            X = np.zeros((n-input_stack+1,dx*input_stack))
            for i in range(input_stack,n+1,1):
                U[i-input_stack,:] = np.flip(new_data[i-input_stack:i,9:13],axis=0).reshape(1,-1)
                X[i-input_stack,:] = np.flip(new_data[i-input_stack:i,:9],axis=0).reshape(1,-1)

            if delta_state:
            # Starts after the data that has requesit U values
                dX = X[1:,:dx]-X[:-1,:dx]
                X = X[:-1, :]
                U = U[:-1, :]
                if battery:
                    batt = np.array(new_data[input_stack-1:-1,-1, None])
                    X = np.hstack((X,batt))

                Time = new_data[input_stack-1:,13]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = new_data[input_stack-1:-1,14]
                Time = Time[:-1]
            else:   # next state predictions
                dX = X[1:,:dx]#-X[:-1,:]
                X = X[:-1,:]
                U = U[:-1,:]
                if battery:
                    batt = np.array(new_data[input_stack-1:-1,-1, None])
                    X = np.hstack((X,batt))
                Time = new_data[input_stack-1:,13]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = new_data[input_stack-1:-1,14]
                Time = Time[:-1]

        ###########################################################################

        else:
            if delta_state:
                X = new_data[1:-2,:9]
                U = new_data[1:-2, 9:13]
                if battery:
                    batt = new_data[1:-2,-1,None]
                    X = np.hstack((X,batt))
                Time = new_data[1:-2,13]
                Objv = new_data[1:-2,14]

                # Reduces by length one for training
                dX = X[1:,:]-X[:-1,:]
                X = X[:-1,:]
                U = U[:-1,:]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = Objv[:-1]
                Time = Time[:-1]
            else:
                X = new_data[1:-2,:9]
                U = new_data[1:-2, 9:13]
                if battery:
                    batt = new_data[1:-2,-1,None]
                    X = np.hstack((X,batt))
                Time = new_data[1:-2,13]
                Objv = new_data[1:-2,14]

                # Reduces by length one for training
                dX = X[1:,:]#-X[:-1,:]
                X = X[:-1,:]
                U = U[:-1,:]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = Objv[:-1]
                Time = Time[:-1]

        ###########################################################################

        # trim some points from takeoff is so desired
        if takeoff_points > 0 and not fastLog:
            takeoff_num = takeoff_points
            X = X[takeoff_num:,:]
            U = U[takeoff_num:,:]
            dX = dX[takeoff_num:,:]
            Objv = Objv[takeoff_num:]
            Ts = Ts[takeoff_num:]
            Time = Time[takeoff_num:]

        ###########################################################################

        if (bound_inputs != []):

            low_bound = bound_inputs[0]
            up_bound = bound_inputs[1]

            # Remove data where U = 0
            X = X[np.array(np.all(U !=0, axis=1)),:]
            dX = dX[np.array(np.all(U !=0, axis=1)),:]
            Objv = Objv[np.array(np.all(U !=0, axis=1))]
            Ts = Ts[np.array(np.all(U !=0, axis=1))]
            Time = Time[np.array(np.all(U !=0, axis=1))]
            U = U[np.array(np.all(U !=0, axis=1)),:]

            # # Remove other values
            Uflag = ~(
                (U[:,0] > up_bound) |
                (U[:,1] > up_bound) |
                (U[:,2] > up_bound) |
                (U[:,3] > up_bound) |
                (U[:,0] < low_bound) |
                (U[:,1] < low_bound) |
                (U[:,2] < low_bound) |
                (U[:,3] < low_bound)
            )
            # print(Uflag)
            X = X[Uflag,:]
            U = U[Uflag,:]
            dX = dX[Uflag,:]
            Objv = Objv[Uflag]
            Ts = Ts[Uflag]
            Time = Time[Uflag]

        ###########################################################################
        # timestep flag of 0 removes points where a 0 timestep is recorded.
        #   looks for data where all timesteps are 0. Can change true to false if
        #   that is so. Then removes all points higher than the second point
        if timestep_flags != []:
            for trim in timestep_flags:
                if np.mean(Ts) < 1:
                    print('~NOTE: heavy trimming may occur, timestamps may be corrupted')
                if trim == 0 and True:
                    # Remove data where Ts = 0
                    X = X[np.array(np.where(Ts > 1)).flatten(),:]
                    U = U[np.array(np.where(Ts > 1)).flatten(),:]
                    dX = dX[np.array(np.where(Ts > 1)).flatten(),:]
                    Objv = Objv[np.array(np.where(Ts > 1)).flatten()]
                    Ts = Ts[np.array(np.where(Ts > 1)).flatten()]
                    Time = Time[np.array(np.where(Ts > 1)).flatten()]
                else:
                    # Remove data where the timestep is wrong
                    # Remove data if timestep above 10ms
                    X = X[np.array(np.where(Ts < trim)).flatten(),:]
                    U = U[np.array(np.where(Ts < trim)).flatten(),:]
                    dX = dX[np.array(np.where(Ts < trim)).flatten(),:]
                    Objv = Objv[np.array(np.where(Ts < trim)).flatten()]
                    Ts = Ts[np.array(np.where(Ts < trim)).flatten()]
                    Time = Time[np.array(np.where(Ts < trim)).flatten()]

        ###########################################################################

        # for if the data may include collisions. Check to match this with the
        #   emergency off command when you were collecting data
        if collision_flag and delta_state:
            # Remove all data for a set of flags
            # YPR step in (-7.5,7.5) deg
            # omega step in (-100,100) deg/s^2
            # accel step in (-10,10) m.s^2
            # STATE FLAGS

            #Create flag for collisions!
            collision_flag = (
                ((X[:,6] < -8)) |
                ((X[:,7] < -8)) |
                ((X[:,8] < -8)) |
                (abs(dX[:,0]) > 75) |
                (abs(dX[:,1]) > 75) |
                (abs(dX[:,2]) > 75)
            )

            if len(np.where(collision_flag==True)[0])>0:
                idx_coll1 = min(np.where(collision_flag==True)[0])
            else:
                idx_coll1 = len(Ts)

            X = X[:idx_coll1,:]
            dX = dX[:idx_coll1,:]
            Objv = Objv[:idx_coll1]
            Ts = Ts[:idx_coll1]
            Time = Time[:idx_coll1]
            U = U[:idx_coll1,:]

        ###########################################################################

        # trims large change is state as we think they are non-physical and a
        #   result of the sensor fusion. Note, this could make prediction less stable
        if trime_large_dX and delta_state:
            # glag = (
            #     ((dX[:,0] > -40) & (dX[:,0] < 40)) &
            #     ((dX[:,1] > -40) & (dX[:,1] < 40)) &
            #     ((dX[:,2] > -40) & (dX[:,2] < 40)) &
            #     ((dX[:,3] > -10) & (dX[:,3] < 10)) &
            #     ((dX[:,4] > -10) & (dX[:,4] < 10)) &
            #     ((dX[:,5] > -10) & (dX[:,5] < 10)) &
            #     ((dX[:,6] > -8) & (dX[:,6] < 8)) &
            #     ((dX[:,7] > -8) & (dX[:,7] < 8)) &
            #     ((dX[:,8] > -8) & (dX[:,8] < 8))
            # )
            glag = (
                ((dX[:,3] > -7.5) & (dX[:,3] < 7.5)) &
                ((dX[:,4] > -7.5) & (dX[:,4] < 7.5)) &
                ((dX[:,5] > -7.5) & (dX[:,5] < 7.5)) &
                ((dX[:,6] > -8) & (dX[:,6] < 8)) &
                ((dX[:,7] > -8) & (dX[:,7] < 8)) &
                ((dX[:,8] > -8) & (dX[:,8] < 8))
            )
            #
            X = X[glag,:]
            dX = dX[glag,:]
            Objv = Objv[glag]
            Ts = Ts[glag]
            Time = Time[glag]
            U = U[glag,:]



        ###########################################################################
        # removes tuples with 0 change in an angle (floats should surely always change)
        if trim_0_dX and delta_state:
            Objv = Objv[np.all(dX[:,3:6] != 0, axis=1)]
            Ts = Ts[np.all(dX[:,3:6] !=0, axis=1)]
            Time = Time[np.all(dX[:,3:6] !=0, axis=1)]
            X = X[np.all(dX[:,3:6] !=0, axis=1)]
            U = U[np.all(dX[:,3:6] !=0, axis=1)]
            dX = dX[np.all(dX[:,3:6] !=0, axis=1)]
        ###########################################################################

        # We do this again when training.
        if shuffle_here:
            # SHUFFLES DATA
            shuff = np.random.permutation(len(Time))
            X = X[shuff,:]
            dX = dX[shuff,:]
            Objv = Objv[shuff]
            Ts = Ts[shuff]
            Time = Time[shuff]
            U = U[shuff,:]

        if find_move:
            # move_idx = np.argmax(np.all(dX[:,3:5] > 0.005, axis=1))
            move_idx = np.argmax(Objv != -1)
            move_idx = int(2*move_idx/3)



        ###########################################################################

        # Can be used to plot trimmed data
        if False:
            font = {'size'   : 18}

            matplotlib.rc('font', **font)
            matplotlib.rc('lines', linewidth=2.5)

            # plt.tight_layout()

            with sns.axes_style("darkgrid"):
                ax1 = plt.subplot(311)
                ax2 = plt.subplot(312)
                ax3 = plt.subplot(313)


            ax1.plot(X[:,3:5])
            ax2.plot(U[:,:4])
            ax3.plot(X[:,6:9])
            plt.show()

        # Make time counting up from first point
        if len(Time) > 0:
            Time -= min(Time[move_idx:])
            Time /= 1000000


        # end of traj marker
        terminals = np.zeros(len(Time))
        if len(terminals)>0: terminals[-1] = 1

        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time), terminals

def df_to_training(df, data_params):
    '''
    Takes in a loaded and trimmed dataframe and a set of (future) parameters to
    train the neural net on. Can take in many dataframes at once
    '''

    # Grab data params
    battery = data_params['battery']
    states = data_params['states']
    inputs = data_params['inputs']
    targets = data_params['targets']


    # dataframe info
    cols = list(df.columns.values) # or list(df)

    # if nothing given, returns all. Old code below.
    if states == [] and inputs == []:
        xu_cols = cols[12:]
        if 'term' in xu_cols: xu_cols.remove('term')
        num_repeat = int((len(xu_cols)-1)/13)+1
        if battery: num_repeat -=1

        dX = df.loc[:,cols[:9]].values
        X = df.loc[:,xu_cols[:9*num_repeat]].values
        U = df.loc[:,xu_cols[9*num_repeat:]].values

    # Otherwise take lists
    else:
        dX = df[targets].values
        X = df[states].values
        U = df[inputs].values

    # NOTE: this makes battery part of the inputs. This is okay, but was originally uninteded
    #   It's okay because the inputs U are scaled by uniform scalers.
    # battery = data_params['battery']
    # if battery:
    #     X = np.hstack((X, df.loc[:,[xu_cols[-1]]].values))

    # TODO: make it possible to choose specific states

    return X, U, dX

def load_dirs(dir_list, load_params):
    df = []
    first = True
    for dir in dir_list:
        df_t = stack_dir_pd(dir, load_params)
        if first:
            df = df_t
            first = False
        else:
            df = df.append(df_t, ignore_index=True)
    print('Processed data of shape: ', df.shape)
    return df

def stack_dir_pd_iono(dir, load_params):
    '''
    Takes in a directory and returns a dataframe for the data, specifically for ionocraft data
    '''
    print('Loading dir: ', dir)
    files = os.listdir("_logged_data_autonomous/iono/"+dir)
    print('...number of flights: ', len(files))

    # init arrays
    X = []
    U = []
    dX = []


    terminals = []

    # init if needed. This will play with terminals a little bit
    if load_params['include_tplus1']:
        tplus1 = []

    for i,f in enumerate(files):
        
        if f[:3] != '.DS':
            # print(f)
            X_t, U_t, dX_t = load_iono_txt(dir+f, load_params)
  
            # shortens length by one point
            if load_params['include_tplus1']:
                if X == []:
                    tplus1 = X_t[1:, :]
                else:
                    tplus1 = np.append(tplus1, X_t[1:, :], axis=0)

                X_t = X_t[:-1, :]
                U_t = U_t[:-1, :]
                dX_t = dX_t[:-1, :]



        
            # State data
            if X == []:
                X = X_t
            else:
                X = np.append(X, X_t, axis=0)

            # inputs
            if U == []:
                U = U_t
            else:
                U = np.append(U, U_t, axis=0)

            # change in state
            if dX == []:
                dX = dX_t
            else:
                dX = np.append(dX, dX_t, axis=0)


            # # end of trajectory marker
            # if terminals == []:
            #     terminals = terminal
            # else:
            #     terminals = np.append(terminals, terminal, axis=0)

    print('...has additional trimmed datapoints: ', np.shape(X)[0])

    ######################################################################

    if False:
        font = {'size': 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', linewidth=2.5)

        # plt.tight_layout()

        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot(111)

        ax1.hist(Ts, bins=250)
        ax1.set_title("Hist of time between log packets")
        ax1.set_xlabel("Time Difference Between Packets (ms)")
        ax1.set_ylabel("Occurence")
        plt.show()

    # Start dataframe

    stack_states = load_params['stack_states']
    if stack_states > 0:
        state_idxs = np.arange(0, 9*stack_states, 9)
        input_idxs = np.arange(0, 4*stack_states, 4)

        d = {'d_omega_x': dX[:, 3],
             'd_omega_y': dX[:, 4],
             'd_omega_z': dX[:, 5],
             'd_pitch': dX[:, 6],
             'd_roll': dX[:, 7],
             'd_yaw': dX[:, 8],
             'd_lina_x': dX[:, 1],
             'd_lina_y': dX[:, 2],
             'd_liny_z': dX[:, 3]
             }

        k = 0
        for i in state_idxs:
            st = str(k)
            k += 1
            d['omega_x'+st] = X[:, 3+i]
            d['omega_y'+st] = X[:, 4+i]
            d['omega_z'+st] = X[:, 5+i]
            d['pitch'+st] = X[:, 6+i]
            d['roll'+st] = X[:, 7+i]
            d['yaw'+st] = X[:, 8+i]
            d['lina_x'+st] = X[:, 0+i]
            d['lina_y'+st] = X[:, 1+i]
            d['lina_z'+st] = X[:, 2+i]

        k = 0
        for j in input_idxs:
            st = str(k)
            k += 1
            d['m1_pwm_'+st] = U[:, 0+j]
            d['m2_pwm_'+st] = U[:, 1+j]
            d['m3_pwm_'+st] = U[:, 2+j]
            d['m4_pwm_'+st] = U[:, 3+j]

    else:  # standard
        d = {'omega_x': X[:, 3],
             'omega_y': X[:, 4],
             'omega_z': X[:, 5],
             'pitch': X[:, 6],
             'roll': X[:, 7],
             'yaw': X[:, 8],
             'lina_x': X[:, 0],
             'lina_y': X[:, 1],
             'liny_z': X[:, 2],

             'm1_pwm': U[:, 0],
             'm2_pwm': U[:, 1],
             'm3_pwm': U[:, 2],
             'm4_pwm': U[:, 3],

             'd_omega_x': dX[:, 3],
             'd_omega_y': dX[:, 4],
             'd_omega_z': dX[:, 5],
             'd_pitch': dX[:, 6],
             'd_roll': dX[:, 7],
             'd_yaw': dX[:, 8],
             'd_lina_x': dX[:, 0],
             'd_lina_y': dX[:, 1],
             'd_lina_z': dX[:, 2]
             }

    # if including tplus 1 (for predicting some raw next states rather than change)
    if load_params['include_tplus1']:
        d['t1_omega_x'] = tplus1[:, 3]
        d['t1_omega_y'] = tplus1[:, 4]
        d['t1_omega_z'] = tplus1[:, 5]
        d['t1_pitch'] = tplus1[:, 6]
        d['t1_roll'] = tplus1[:, 7]
        d['t1_yaw'] = tplus1[:, 8]
        d['t1_lina_x'] = tplus1[:, 0]
        d['t1_lina_y'] = tplus1[:, 1]
        d['t1_lina_z'] = tplus1[:, 2]

    df = pd.DataFrame(data=d)
    # print(df)
    print('\n')
    return df


def dir_summary_csv(dir, load_params):
    # takes in a directory with loading parameters and saves a csv summarizing each flight
    print('-------------------')
    print('Loading dir: ', dir)
    files = os.listdir("_logged_data_autonomous/_newquad1/publ2/"+dir)
    # files = os.listdir(dir)
    # print('...number of flights: ', len(files))

    # init arrays
    X = []
    U = []
    dX = []
    objv = []
    Ts = []
    times = []
    terminals = []

    save_dir = "_summaries/"
    end_idx = dir[-2::-1].find('/')
    saved_name = save_dir + "summary-" + dir[-end_idx-1:]+'.csv'
    print(dir)
    # print(saved_name)
    with open(saved_name, 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=',')
        writer.writerow(["Flight Idx", "Flight Time (ms)", "Trainable Points", "Mean Objective", "RMS Pitch Roll"])
        for i,f in enumerate(files):
            print(f)
            if len(f) > 5 and f[-4:] == '.csv':
                X_t, U_t, dX_t, objv_t, Ts_t, time, terminal = trim_load_param("_logged_data_autonomous/_newquad1/publ2/"+dir+"/"+f, load_params)

                flight_time = np.round(np.max(time),2)
                mean_obj =  np.round(np.mean(objv_t[objv_t != -1]),2)
                rmse =  np.round(np.sqrt(np.mean(np.sum(X_t[:,3]**2+X_t[:,4]**2))),2)
                num_points = len(time)
                writer.writerow([str(i), str(flight_time), str(num_points), str(mean_obj), str(rmse)])

def rollouts_summary_csv(dir):
    # takes in a directory with loading parameters and saves a csv summarizing each flight
    print('-------------------')
    print('Loading dir: ', dir)
    files = os.listdir(dir)
    # files = os.listdir(dir)
    # print('...number of flights: ', len(files))

    # init arrays

    save_dir = "_summaries/"
    end_idx = dir[-2::-1].find('/')
    saved_name = save_dir + "summary-" + dir[-end_idx-1:-1]+'.csv'
    print(dir)
    # print(saved_name)
    with open(saved_name, 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=',')
        writer.writerow(["Rollout", "Mean Flight Time", "Std Flight Time", "Total Trained Points", "RMS Pitch Roll"])
        for i,f in enumerate(sorted(files)):
            # print(f)
            if len(f) > 5 and f[-4:] == '.csv':
                df = pd.read_csv(dir+"/"+f, sep=",")

                flight_time_mean = np.round(np.mean(df["Flight Time (ms)"]),2)
                flight_time_std = np.round(np.std(df["Flight Time (ms)"]),2)
                num_points = np.round(np.sum([df["Trained Points"]]),2)
                mean_obj =  np.round(np.mean(df["Mean Objective"]),2)
                rmse =  np.round(np.mean(df["RMS Pitch Roll"]),2)

                writer.writerow([f[-f[::-1].find('_'):-f[::-1].find('.')-1],
                                 str(flight_time_mean),
                                 str(flight_time_std),
                                 str(num_points),
                                 str(rmse)])


def get_rand_traj(df):
    '''
    Given a loaded dataframe, calculates how many trajectories there are and
    returns a random trajectory, with its position
    '''
    if "term" not in list(df.columns.values):
        raise ValueError("Did not have terminal column in dataframe")

    ends = np.squeeze(np.where(df['term'].values==1))
    points = np.concatenate((np.array([0]), ends))

    end_index = np.random.randint(len(ends))
    start, end = points[end_index:end_index+2]
    # print(start)
    df_sub = df[start+1:end+1]
    # print(df_sub)

    return df_sub, end_index

def get_traj(df,idx):
    '''
    Given a loaded dataframe and an index, returns the idx'th tajectory from the
    list. This is useful as a followup once you have gotten a random one you enjoy
    '''
    if "term" not in list(df.columns.values):
        raise ValueError("Did not have terminal column in dataframe")

    ends = np.squeeze(np.where(df['term'].values==1))
    points = np.concatenate((np.array([0]), ends))

    end_index = idx
    start, end = points[end_index:end_index+2]
    # print(start)
    df_sub = df[start+1:end+1]

    return df_sub

def load_iono_txt(fname, load_params):
    """
    This fnc will read and parse the data from an ionocraft flight towards the same format of (X,U, dX).
    - Will return a df here
    - Will add plotting functionality

    The raw file has lines from Arduino serial print of the form:
    pwm1, pwm2, pwm3, pwm4, ax, ay, az, wx, wy, wz, pitch, roll, yaw
    """

    # Grab params
    delta_state = load_params['delta_state']
    include_tplus1 = load_params['include_tplus1']
    takeoff_points = load_params['takeoff_points']
    trim_0_dX = load_params['trim_0_dX']
    trime_large_dX = load_params['trime_large_dX']
    find_move = load_params['find_move']
    input_stack = load_params['stack_states']
    shuffle_here = False #load_params['shuffle_here']
    battery = False
    zero_yaw = load_params['zero_yaw']
    m_avg = int(load_params['moving_avg'])

    # files = os.listdir("_logged_data_autonomous/"+dir)
    file = "_logged_data_autonomous/iono/" + fname
    # print(file)
    with open(file, "rb") as csvfile:
        # laod data
        # cols_use = np.linspace(0,13,14)
        print(csvfile)
        cols_use = (0,1,2,3,4,5,6,7,8,9,10,11,12)
        new_data = np.genfromtxt(csvfile, delimiter=",", usecols=cols_use, autostrip=True)
        print(new_data)
        # print(new_data.shape)

        

        serial_error_flag = (
            ((new_data[:, -1] > -360) & (new_data[:, -1] < 360)) &  # yaw
            ((new_data[:, -2] > -360) & (new_data[:, -2] < 360)) &  # roll
            ((new_data[:, -3] > -360) & (new_data[:, -3] < 360)) &  # pitch
            ((new_data[:, 4] > -500) & (new_data[:, 4] < 500)) &
            ((new_data[:, 5] > -500) & (new_data[:, 5] < 500)) &
            ((new_data[:, 6] > -500) & (new_data[:, 6] < 500))
        )
        
    
        new_data = new_data[serial_error_flag,:]

        if True and m_avg > 1:
            # fitlers the euler angles by targeted value
            new_data[:, -1] = np.convolve(
                new_data[:, -1], np.ones((m_avg,))/m_avg, mode='same')
            new_data[:, -2] = np.convolve(
                new_data[:, -2], np.ones((m_avg,))/m_avg, mode='same')
            new_data[:, -3] = np.convolve(
                new_data[:, -3], np.ones((m_avg,))/m_avg, mode='same')

            # filters accelerations by 2
            new_data[:, 4] = np.convolve(
                new_data[:, 4], np.ones((2,))/2, mode='same')
            new_data[:, 5] = np.convolve(
                new_data[:, 5], np.ones((2,))/2, mode='same')
            new_data[:, 6] = np.convolve(
                new_data[:, 6], np.ones((2,))/2, mode='same')
        
        # Check the range for each val
        if False:
            for c in cols_use:
                print("State col: ", c)
                print("Max val: ", np.max(new_data[:,c]))
                print("Min val: ", np.min(new_data[:, c]))
                print("Mean val: ", np.mean(new_data[:, c]))
                print('------')

    
        # TODO: Modify this code so it matches what we have here rather than the CF stuff
        ########### THESE BARS SEPARATE TRIMMING ACTIONS #########################
        # For now, remove the last 4 columns becausee they're PWMS

        # add pwm latency calculations
        pwm_rec = new_data[:, 0:4]

        ###########################################################################
        # adding to make the input horizontally stacked set of inputs, rather than only the last input because of spinup time
        if input_stack > 1:
            n, du = np.shape(new_data[:, 0:4])
            _, dx = np.shape(new_data[:, 4:])

            U = np.zeros((n-input_stack+1, du*input_stack))
            X = np.zeros((n-input_stack+1, dx*input_stack))
            for i in range(input_stack, n+1, 1):
                U[i-input_stack,
                    :] = np.flip(new_data[i-input_stack:i, 0:4], axis=0).reshape(1, -1)
                X[i-input_stack,
                    :] = np.flip(new_data[i-input_stack:i, 4:], axis=0).reshape(1, -1)
            
            if delta_state:
                # Starts after the data that has requesit U values
                dX = X[1:, :dx]-X[:-1, :dx]
                X = X[:-1, :]
                U = U[:-1, :]

            else:   # next state predictions
                dX = X[1:, :dx]  # -X[:-1,:]
                X = X[:-1, :]
                U = U[:-1, :]
            
            if zero_yaw:
                # Need to change to correct dimension here
                X[:, 8] = X[:, 8]-X[0, 8]
       
        else:
            n, du = np.shape(new_data[:, 0:4])
            _, dx = np.shape(new_data[:, 4:])

            U = np.zeros((n-input_stack+1, du*input_stack))
            X = np.zeros((n-input_stack+1, dx*input_stack))

            if delta_state:
                # Starts after the data that has requesit U values
                dX = X[1:, :dx]-X[:-1, :dx]
                X = X[:-1, :]
                U = U[:-1, :]

            else:   # next state predictions
                dX = X[1:, :dx]  # -X[:-1,:]
                X = X[:-1, :]
                U = U[:-1, :]

            if zero_yaw:
                # Need to change to correct dimension here
                X[:, 8] = X[:, 8]-X[0, 8]

        # print("State data shape, ", X.shape)
        # print("Input data shape, ", U.shape)
        # print("Change state data shape, ", dX.shape)
        
        if trim_0_dX and delta_state:
            X = X[np.all(dX[:, 6:] != 0, axis=1)]
            U = U[np.all(dX[:, 6:] != 0, axis=1)]
            dX = dX[np.all(dX[:, 6:] != 0, axis=1)]
        
        # trims large change is state as we think they are non-physical and a
        #   result of the sensor fusion. Note, this could make prediction less stable
        if trime_large_dX and delta_state:
            glag = (
                ((dX[:, 3] > -7.5) & (dX[:, 3] < 7.5)) &
                ((dX[:, 4] > -7.5) & (dX[:, 4] < 7.5)) &
                ((dX[:, 5] > -7.5) & (dX[:, 5] < 7.5)) &
                ((dX[:, 6] > -8) & (dX[:, 6] < 8)) &
                ((dX[:, 7] > -8) & (dX[:, 7] < 8)) &
                ((dX[:, 8] > -8) & (dX[:, 8] < 8))
            )
            
            #
            X = X[glag, :]
            dX = dX[glag, :]
            U = U[glag, :]

            dX = X[1:, :dx]-X[:-1, :dx]
            X = X[:-1, :]
            U = U[:-1, :]
            glag = (
                ((dX[:, 3] > -7.5) & (dX[:, 3] < 7.5)) &
                ((dX[:, 4] > -7.5) & (dX[:, 4] < 7.5)) &
                ((dX[:, 5] > -7.5) & (dX[:, 5] < 7.5)) &
                ((dX[:, 6] > -8) & (dX[:, 6] < 8)) &
                ((dX[:, 7] > -8) & (dX[:, 7] < 8)) &
                ((dX[:, 8] > -8) & (dX[:, 8] < 8))
            )

            #
            X = X[glag, :]
            dX = dX[glag, :]
            U = U[glag, :]

            dX = X[1:, :dx]-X[:-1, :dx]
            X = X[:-1, :]
            U = U[:-1, :]
            glag = (
                ((dX[:, 3] > -7.5) & (dX[:, 3] < 7.5)) &
                ((dX[:, 4] > -7.5) & (dX[:, 4] < 7.5)) &
                ((dX[:, 5] > -7.5) & (dX[:, 5] < 7.5)) &
                ((dX[:, 6] > -8) & (dX[:, 6] < 8)) &
                ((dX[:, 7] > -8) & (dX[:, 7] < 8)) &
                ((dX[:, 8] > -8) & (dX[:, 8] < 8))
            )

            #
            X = X[glag, :]
            dX = dX[glag, :]
            U = U[glag, :]
            
            # this is repeated three times for some anomolous serial data
       

        if True:
            # font = {'size': 22,'family': 'serif', 'serif': ['Times']}
            font = {'size': 12}

            matplotlib.rc('font', **font)
            matplotlib.rc('lines', linewidth=1.25)
            matplotlib.rc('text', usetex=True)

            # plt.tight_layout()

            fig = plt.figure()
            with sns.axes_style("whitegrid"):
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["axes.edgecolor"] = "0.15"
                plt.rcParams["axes.linewidth"] = 1.5
                plt.subplots_adjust(top=.93, bottom=0.13, left=.13,
                                    right=1-.03, hspace=.25)
                ax1 = plt.subplot(111)
                

            ax1.set_title("Logged Iono Data")
            ax1.set_xlabel("Datapoint (Collected Approx. 200Hz)")
            ax1.set_ylabel("Angle (Degrees)")

            if False:
                def butter_lowpass(cutoff, fs, order=5):
                    nyq = 0.5 * fs
                    normal_cutoff = cutoff / nyq
                    b, a = butter(order, normal_cutoff, btype='low', analog=False)
                    return b, a


                def butter_lowpass_filter(data, cutoff, fs, order=5):
                    b, a = butter_lowpass(cutoff, fs, order=order)
                    y = lfilter(b, a, data)
                    return y
                
                # Filter requirements.
                order = 6
                fs = 100.0       # sample rate, Hz
                cutoff = 1.667  # desired cutoff frequency of the filter, Hz
                
                # Get the filter coefficients so we can check its frequency response.
                b, a = butter_lowpass(cutoff, fs, order)

                ax1.plot(butter_lowpass_filter(
                    X[:, 6], cutoff, fs, order), label='filtered pitch')
                ax1.plot(butter_lowpass_filter(
                    X[:, 7], cutoff, fs, order), label='filtered roll')
                ax1.plot(butter_lowpass_filter(
                    X[:, 8], cutoff, fs, order), label='filtered yaw')
            else:   
                ax1.plot(X[:, -3], label='roll')
                ax1.plot(X[:, -2], label='pitch')
                ax1.plot(X[:, -1], label='yaw')

                # ax1.plot(U[:200, 0], label='T1')
                # ax1.plot(U[:200, 1], label='T2')
                # ax1.plot(U[:200, 2], label='T3')
                # ax1.plot(U[:200, 3], label='T4')

            ax1.legend()
            plt.show()
            fig.set_size_inches(8, 4.5)

            # plt.savefig('psoter', edgecolor='black', dpi=100, transparent=True)

            plt.savefig('iono_flight.eps', format='eps', dpi=300)
            # quit()

        # print("State data shape, ", X.shape)
        # print("Input data shape, ", U.shape)
        # print("Change state data shape, ", dX.shape)

        return X, U, dX
