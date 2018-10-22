
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

    for f in files:
        # print(f)
        X_t, U_t, dX_t, objv_t, Ts_t, time, terminal = trim_load_param("_logged_data_autonomous/"+dir+f, load_params)

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
            d['liny_z'+st] = X[:,8+i]

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
            'd_liny_z': dX[:,8],

            'timesteps': Ts[:],
            'objective vals': objv[:],
            'flight times': times[:]
            }

    track_terminals = load_params['stack_states']
    if track_terminals: d['term'] = terminals

    # loads battery if needed
    battery = load_params['battery']
    if battery: d['vbat'] = X[:,-1]

    df = pd.DataFrame(data=d)
    # print(df)
    print('\n')
    return df

def trim_load_param(fname, load_params):
    '''
    Opens the directed csv file and returns the arrays we want
    '''

    # Grab params
    delta_state = load_params['delta_state']
    takeoff_points = load_params['takeoff_points']
    trim_0_dX = load_params['trim_0_dX']
    trime_large_dX = load_params['trime_large_dX']
    bound_inputs = load_params['bound_inputs']
    input_stack = load_params['stack_states']
    collision_flag = load_params['collision_flag']
    shuffle_here = load_params['shuffle_here']
    timestep_flags = load_params['timestep_flags']
    battery = load_params['battery']

    with open(fname, "rb") as csvfile:
        # laod data
        new_data = np.loadtxt(csvfile, delimiter=",")

        ########### THESE BARS SEPARATE TRIMMING ACTIONS #########################

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
        if takeoff_points > 0:
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
            glag = (
                ((dX[:,0] > -40) & (dX[:,0] < 40)) &
                ((dX[:,1] > -40) & (dX[:,1] < 40)) &
                ((dX[:,2] > -40) & (dX[:,2] < 40)) &
                ((dX[:,3] > -6) & (dX[:,3] < 6)) &
                ((dX[:,4] > -6) & (dX[:,4] < 6)) &
                ((dX[:,5] > -6) & (dX[:,5] < 6)) &
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
            Objv = Objv[np.all(dX[:,3:6] !=0, axis=1)]
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

        ###########################################################################

        # Can be used to plot trimmed data
        if False:
            font = {'size'   : 18}

            matplotlib.rc('font', **font)
            matplotlib.rc('lines', linewidth=2.5)

            # plt.tight_layout()

            with sns.axes_style("darkgrid"):
                ax1 = plt.subplot(211)
                ax2 = plt.subplot(212)


            ax1.plot(X[:,3:6])
            ax2.plot(U[:,:4])
            plt.show()

        # Make time counting up from first point
        if len(Time) > 0:
            Time -= min(Time)
            Time /= 1000000

        # end of traj marker
        terminals = np.zeros(len(Time))
        terminals[-1] = 1

        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time)

def df_to_training(df, data_params):
    '''
    Takes in a loaded and trimmed dataframe and a set of (future) parameters to
    train the neural net on. Can take in many dataframes at once
    '''
    battery = data_params['battery']

    cols = list(df.columns.values) # or list(df)

    xu_cols = cols[12:]
    num_repeat = int((len(xu_cols)-1)/13)+1
    if battery: num_repeat -=1

    dX = df.loc[:,cols[:9]].values
    X = df.loc[:,xu_cols[:9*num_repeat]].values
    U = df.loc[:,xu_cols[9*num_repeat:]].values

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
