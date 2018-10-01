
# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datetime import timedelta
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd

def stack_dir_pd(dir, load_params):
    '''
    Takes in a directory and saves the compiled tests into one numpy .npz in
    _logged_data_autonomous/compiled/
    '''
    files = os.listdir("_logged_data_autonomous/"+dir)
    print('Number of flights: ', len(files))

    # init arrays
    X = []
    U = []
    dX = []
    objv = []
    Ts = []
    times = []

    for f in files:
        # print(f)
        X_t, U_t, dX_t, objv_t, Ts_t, time = trim_load_param("_logged_data_autonomous/"+dir+f, load_params)

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

    print('Directory: ', dir, ' has additional trimmed datapoints: ', np.shape(X)[0])

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


    df = pd.DataFrame(data=d)
    print(df)
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
                Time = new_data[input_stack-1:,13]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = new_data[input_stack-1:-1,14]
                Time = Time[:-1]
            else:   # next state predictions
                dX = X[1:,:dx]#-X[:-1,:]
                X = X[:-1,:]
                U = U[:-1,:]
                Time = new_data[input_stack-1:,13]
                Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
                Objv = new_data[input_stack-1:-1,14]
                Time = Time[:-1]

        ###########################################################################

        else:
            if delta_state:
                X = new_data[1:-2,:9]
                U = new_data[1:-2, 9:13]
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

        # Make time counting up from first point
        if len(Time) > 0:
            Time -= min(Time)
            Time /= 1000000

        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time)

def stack_dir(dir, delta, input_stack = 0, takeoff=False):
    '''
    Takes in a directory and saves the compiled tests into one numpy .npz in
    _logged_data_autonomous/compiled/
    '''
    files = os.listdir("_logged_data_autonomous/"+dir)
    print('Number of flights: ', len(files))
    X = []
    U = []
    dX = []
    Ts = []
    times = []
    for f in files:
        # print(f)
        # with open(f, "rb") as csvfile:
        #     new_data = np.loadtxt(csvfile, delimiter=",")
        if delta:
            X_t, U_t, dX_t, _, Ts_t, time = trim_load_delta("_logged_data_autonomous/"+dir+f, input_stack = input_stack, takeoff=takeoff)
            if times == []:
                times = time
            else:
                times = np.append(times,time)
        else:
            X_t, U_t, dX_t, _, Ts_t, _ = trim_load_next("_logged_data_autonomous/"+dir+f, action_avg = input_stack)

        if X == []:
            X = X_t
        else:
            X = np.append(X, X_t, axis=0)

        if U == []:
            U = U_t
        else:
            U = np.append(U, U_t, axis=0)

        if dX == []:
            dX = dX_t
        else:
            dX = np.append(dX, dX_t, axis=0)

        if Ts_t == []:
            Ts = Ts_t
        else:
            Ts = np.append(Ts, Ts_t, axis=0)

    print('Average flight length is: ', np.mean(times))
    print('Mean Squared Roll + Pitch:', np.mean(X[:,3]**2+X[:,4]**2))
    # plt.figure()
    # plt.hist(Ts, bins=100)
    # plt.show()
    # quit()
    print('Directory: ', dir, ' has additional trimmed datapoints: ', np.shape(X)[0])
    return np.array(X), np.array(U), np.array(dX)

def trim_load_next(fname, action_avg = 0):
    '''
    Opens the directed csv file and returns the arrays we want
    '''
    with open(fname, "rb") as csvfile:
        new_data = np.loadtxt(csvfile, delimiter=",")

        # adding to make the input horizontally stacked set of inputs, rather than only the last input because of spinup time
        if action_avg >1:
            n, du = np.shape(new_data[:,9:13])
            U = np.zeros((n-action_avg+1,du*action_avg))
            for i in range(action_avg,n+1,1):
                U[i-action_avg,:] = new_data[i-action_avg:i,9:13].reshape(1,-1)

            # Starts after the data that has requesit U values
            X = new_data[action_avg-1:-2,:9]
            U = U[:-2, :]
            Time = new_data[action_avg-1:-2,13]
            Objv = new_data[action_avg-1:-2,14]
        else:
            # Starts after 20th data point because MANY repeats
            X = new_data[1:-2,:9]
            U = new_data[1:-2, 9:13]
            Time = new_data[1:-2,13]
            Objv = new_data[1:-2,14]

        # X[:,:3] += np.random.uniform(-0.352112676/2, 0.352112676/2, size =np.shape(X[:,3:6]))
        # X[:,3:6] += np.random.uniform(-0.176056338/2, 0.176056338/2, size =np.shape(X[:,3:6]))
        # X[:,6:] += np.random.uniform(-0.078125/2, 0.078125/2, size =np.shape(X[:,3:6]))

        # Reduces by length one for training
        dX = X[1:,:]#-X[:-1,:]
        X = X[:-1,:]
        U = U[:-1,:]
        Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
        Objv = Objv[:-1]
        Time = Time[:-1]

        # # Remove data where the timestep is wrong
        # # Remove data if timestep above 10ms
        X = X[np.array(np.where(Ts < 9)).flatten(),:]
        U = U[np.array(np.where(Ts < 9)).flatten(),:]
        dX = dX[np.array(np.where(Ts < 9)).flatten(),:]
        Objv = Objv[np.array(np.where(Ts < 9)).flatten()]
        Ts = Ts[np.array(np.where(Ts < 9)).flatten()]
        Time = Time[np.array(np.where(Ts < 9)).flatten()]
        # #
        # # # Remove data where Ts = 0
        X = X[np.array(np.where(Ts > 1)).flatten(),:]
        U = U[np.array(np.where(Ts > 1)).flatten(),:]
        dX = dX[np.array(np.where(Ts > 1)).flatten(),:]
        Objv = Objv[np.array(np.where(Ts > 1)).flatten()]
        Ts = Ts[np.array(np.where(Ts > 1)).flatten()]
        Time = Time[np.array(np.where(Ts > 1)).flatten()]
        #
        #
        # # Remove data where U = 0
        X = X[np.array(np.all(U !=0, axis=1)),:]
        dX = dX[np.array(np.all(U !=0, axis=1)),:]
        Objv = Objv[np.array(np.all(U !=0, axis=1))]
        Ts = Ts[np.array(np.all(U !=0, axis=1))]
        Time = Time[np.array(np.all(U !=0, axis=1))]
        U = U[np.array(np.all(U !=0, axis=1)),:]
        #
        # # Remove all data for a set of flags
        # # YPR step in (-7.5,7.5) deg
        # # omega step in (-100,100) deg/s^2
        # # accel step in (-10,10) m.s^2
        # # STATE FLAGS
        #
        collision_flag = (
            ((X[:,6] < -6)) |
            ((X[:,7] < -6)) |
            ((X[:,8] < -6))
        )
        #
        if len(np.where(collision_flag==True)[0])>0:
            idx_coll1 = min(np.where(collision_flag==True)[0])
        else:
            idx_coll1 = len(Ts)
        # # print(idx_coll1)
        #
        X = X[:idx_coll1,:]
        dX = dX[:idx_coll1,:]
        Objv = Objv[:idx_coll1]
        Ts = Ts[:idx_coll1]
        Time = Time[:idx_coll1]
        U = U[:idx_coll1,:]
        #

        # # FLAG FOR 0 CHANGE IN ANGLE AT HIGH ANGLES
        # glag = ~(
        #     ((abs(X[:,3]) > 12.5) & (dX[:,3] == 0)) |
        #     ((abs(X[:,4]) > 12.5) & (dX[:,4] == 0)) |
        #     ((abs(X[:,0]) > 75) & (dX[:,0] == 0)) |
        #     ((abs(X[:,1]) > 75) & (dX[:,1] == 0)) |
        #     ((abs(X[:,2]) > 75) & (dX[:,2] == 0))
        # )
        #
        # X = X[glag,:]
        # dX = dX[glag,:]
        # Objv = Objv[glag]
        # Ts = Ts[glag]
        # Time = Time[glag]
        # U = U[glag,:]
        #
        #
        #
        # # X = np.delete(X,5,1)      # can be sed to delete YAW
        #
        #
        # # Add noise to quantized data
        # # dX[:,:3] += np.random.uniform(-0.352112676/2, 0.352112676/2, size =np.shape(dX[:,3:6]))
        # # dX[:,3:6] += np.random.uniform(-0.176056338/2, 0.176056338/2, size =np.shape(dX[:,3:6]))
        # # dX[:,6:] += np.random.uniform(-0.078125/2, 0.078125/2, size =np.shape(dX[:,3:6]))
        #
        #
        # # ypr = dX[np.where(np.logical_and(dX[:,6:] > -7.5, dX[:,6:] < 7.5))]
        # # lin = (dX[:,3:6] > -10) and (dX[:,3:6] < 10)
        # # omeg = (dX[:,:3] > -100) and (dX[:,:3] < 100)
        # # if False:
        # #     X = X[np.all(X[:,3:] !=0, axis=1)]
        # #     U = U[np.all(X[:,3:] !=0, axis=1)]
        # #     dX = dX[np.all(X[:,3:] !=0, axis=1)]
        #
        # # SHUFFLES DATA
        # shuff = np.random.permutation(len(Time))
        # X = X[shuff,:]
        # dX = dX[shuff,:]
        # Objv = Objv[shuff]
        # Ts = Ts[shuff]
        # Time = Time[shuff]
        # U = U[shuff,:]


        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time)

def trim_load_delta(fname, input_stack = 0, takeoff=False):
    '''
    Opens the directed csv file and returns the arrays we want
    '''
    with open(fname, "rb") as csvfile:
        new_data = np.loadtxt(csvfile, delimiter=",")

        # adding to make the input horizontally stacked set of inputs, rather than only the last input because of spinup time
        if input_stack >1:
            n, du = np.shape(new_data[:,9:13])
            _, dx = np.shape(new_data[:,:9])
            U = np.zeros((n-input_stack+1,du*input_stack))
            X = np.zeros((n-input_stack+1,dx*input_stack))
            for i in range(input_stack,n+1,1):
                # print(i-input_stack)
                # print(i)
                # print(new_data[i-input_stack:i,9:13])
                # print(np.flip(new_data[i-input_stack:i,9:13] ,axis=0))
                U[i-input_stack,:] = np.flip(new_data[i-input_stack:i,9:13],axis=0).reshape(1,-1)
                X[i-input_stack,:] = np.flip(new_data[i-input_stack:i,:9],axis=0).reshape(1,-1)
                # quit()
            # Starts after the data that has requesit U values
            dX = X[1:,:dx]-X[:-1,:dx]
            dX = dX[:]
            X = X[:-1, :]
            U = U[:-1, :]
            Time = new_data[input_stack-1:,13]
            Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
            Objv = new_data[input_stack-1:-1,14]
            Time = Time[:-1]
        else:
            X = new_data[1:-2,:9]
            U = new_data[1:-2, 9:13]
            Time = new_data[1:-2,13]
            Objv = new_data[1:-2,14]

            # Reduces by length one for training
            dX = X[1:,:]-X[:-1,:]
            X = X[:-1,:]
            U = U[:-1,:]
            Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
            Objv = Objv[:-1]
            Time = Time[:-1]
        # print(np.shape(X))
        # print(np.shape(U))
        # print(np.shape(Time))
        # print(np.shape(Objv))


        # print(np.shape(dX))
        # print(np.shape(X))
        # print(np.shape(U))
        # print(np.shape(Objv))
        # print(np.shape(Ts))
        # print(U)
        # print(X)
        # print(dX)
        # quit()
        # for removing takeoffs

        if takeoff:
            vals = np.where(np.all((U>4000),axis = 1))
            val = np.min(vals)
            takeoff_num =175
            X = X[takeoff_num:,:]
            U = U[takeoff_num:,:]
            dX = dX[takeoff_num:,:]
            Objv = Objv[takeoff_num:]
            Ts = Ts[takeoff_num:]
            Time = Time[takeoff_num:]
        else:
            # # Remove open loop take off values
            Uflag = ~(
                (U[:,0] > 50000) |
                (U[:,1] > 50000) |
                (U[:,2] > 50000) |
                (U[:,3] > 50000)
            )
            # print(Uflag)
            X = X[Uflag,:]
            U = U[Uflag,:]
            dX = dX[Uflag,:]
            Objv = Objv[Uflag]
            Ts = Ts[Uflag]
            Time = Time[Uflag]

        # # Remove data where the timestep is wrong
        # # Remove data if timestep above 10ms
        # X = X[np.array(np.where(Ts < 7)).flatten(),:]
        # U = U[np.array(np.where(Ts < 7)).flatten(),:]
        # dX = dX[np.array(np.where(Ts < 7)).flatten(),:]
        # Objv = Objv[np.array(np.where(Ts < 7)).flatten()]
        # Ts = Ts[np.array(np.where(Ts < 7)).flatten()]
        # Time = Time[np.array(np.where(Ts < 7)).flatten()]
        # #
        # # Remove data where Ts = 0
        # X = X[np.array(np.where(Ts > 1)).flatten(),:]
        # U = U[np.array(np.where(Ts > 1)).flatten(),:]
        # dX = dX[np.array(np.where(Ts > 1)).flatten(),:]
        # Objv = Objv[np.array(np.where(Ts > 1)).flatten()]
        # Ts = Ts[np.array(np.where(Ts > 1)).flatten()]
        # Time = Time[np.array(np.where(Ts > 1)).flatten()]
        #
        #

        #
        # # Remove all data for a set of flags
        # # YPR step in (-7.5,7.5) deg
        # # omega step in (-100,100) deg/s^2
        # # accel step in (-10,10) m.s^2
        # # STATE FLAGS
        #
        # # print(X[:,6:])
        # Create flag for collisions!
        # collision_flag = (
        #     ((X[:,6] < -8)) |
        #     ((X[:,7] < -8)) |
        #     ((X[:,8] < -8)) |
        #     (abs(dX[:,0]) > 75) |
        #     (abs(dX[:,1]) > 75) |
        #     (abs(dX[:,2]) > 75)
        # )
        # #
        # if len(np.where(collision_flag==True)[0])>0:
        #     idx_coll1 = min(np.where(collision_flag==True)[0])
        # else:
        #     idx_coll1 = len(Ts)
        # # print(idx_coll1)
        #
        # X = X[:idx_coll1,:]
        # dX = dX[:idx_coll1,:]
        # Objv = Objv[:idx_coll1]
        # Ts = Ts[:idx_coll1]
        # Time = Time[:idx_coll1]
        # U = U[:idx_coll1,:]

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

        # Remove data where U = 0
        X = X[np.array(np.all(U !=0, axis=1)),:]
        dX = dX[np.array(np.all(U !=0, axis=1)),:]
        Objv = Objv[np.array(np.all(U !=0, axis=1))]
        Ts = Ts[np.array(np.all(U !=0, axis=1))]
        Time = Time[np.array(np.all(U !=0, axis=1))]
        U = U[np.array(np.all(U !=0, axis=1)),:]

        #
        glag = (
            ((dX[:,0] > -50) & (dX[:,0] < 50)) &
            ((dX[:,1] > -50) & (dX[:,1] < 50)) &
            ((dX[:,2] > -50) & (dX[:,2] < 50)) &
            ((dX[:,3] > -5) & (dX[:,3] < 5)) &
            ((dX[:,4] > -5) & (dX[:,4] < 5)) &
            ((dX[:,5] > -5) & (dX[:,5] < 5)) &
            ((dX[:,6] > -5) & (dX[:,6] < 5)) &
            ((dX[:,7] > -5) & (dX[:,7] < 5)) &
            ((dX[:,8] > -5) & (dX[:,8] < 5))
        )
        #
        X = X[glag,:]
        dX = dX[glag,:]
        Objv = Objv[glag]
        Ts = Ts[glag]
        Time = Time[glag]
        U = U[glag,:]

        glag = (
            ((dX[:,5] > -10) & (dX[:,5] < 10))
        )
        #
        X = X[glag,:]
        dX = dX[glag,:]
        Objv = Objv[glag]
        Ts = Ts[glag]
        Time = Time[glag]
        U = U[glag,:]
        #
        # FLAG FOR 0 out high angles to regularize different testing.
        # glag = ~((abs(X[:,3]) > 35) | (abs(X[:,4]) > 35))
        #
        # X = X[glag,:]
        # dX = dX[glag,:]
        # Objv = Objv[glag]
        # Ts = Ts[glag]
        # Time = Time[glag]
        # U = U[glag,:]


        #
        # # X = np.delete(X,5,1)      # can be sed to delete YAW
        #
        #
        # # Add noise to quantized data
        # dX[:,:3] += np.random.uniform(-0.352112676/20, 0.352112676/20, size =np.shape(dX[:,3:6]))
        # dX[:,3:6] += np.random.uniform(-0.176056338/20, 0.176056338/20, size =np.shape(dX[:,3:6]))
        # dX[:,6:] += np.random.uniform(-0.078125/20, 0.078125/20, size =np.shape(dX[:,3:6]))
        #
        # X[:,:3] += np.random.uniform(-0.352112676/20, 0.352112676/20, size =np.shape(dX[:,3:6]))
        # X[:,3:6] += np.random.uniform(-0.176056338/20, 0.176056338/20, size =np.shape(dX[:,3:6]))
        # X[:,6:] += np.random.uniform(-0.078125/20, 0.078125/20, size =np.shape(dX[:,3:6]))
        #
        # U[:,:3] += np.random.uniform(-0.352112676/20, 0.352112676/20, size =np.shape(dX[:,3:6]))
        # U[:,3:6] += np.random.uniform(-0.176056338/20, 0.176056338/20, size =np.shape(dX[:,3:6]))
        # U[:,6:] += np.random.uniform(-0.078125/20, 0.078125/20, size =np.shape(dX[:,3:6]))
        #
        #
        # # ypr = dX[np.where(np.logical_and(dX[:,6:] > -7.5, dX[:,6:] < 7.5))]
        # # lin = (dX[:,3:6] > -10) and (dX[:,3:6] < 10)
        # # omeg = (dX[:,:3] > -100) and (dX[:,:3] < 100)
        if True:
            Objv = Objv[np.all(dX[:,3:6] !=0, axis=1)]
            Ts = Ts[np.all(dX[:,3:6] !=0, axis=1)]
            Time = Time[np.all(dX[:,3:6] !=0, axis=1)]
            X = X[np.all(dX[:,3:6] !=0, axis=1)]
            U = U[np.all(dX[:,3:6] !=0, axis=1)]
            dX = dX[np.all(dX[:,3:6] !=0, axis=1)]

        #
        # SHUFFLES DATA
        # shuff = np.random.permutation(len(Time))
        # X = X[shuff,:]
        # dX = dX[shuff,:]
        # Objv = Objv[shuff]
        # Ts = Ts[shuff]
        # Time = Time[shuff]
        # U = U[shuff,:]
        # plt.plot(X[:,3:6])
        # plt.plot(U[:,:])
        # plt.show()
        flight_len = 0
        if len(Ts) > 0:
            flight_len = (max(Time)-min(Time))/1000000
        if True:
            Time -= min(Time)
            Time /= 1000000
        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time)

# returns the elapsed milliseconds since the start of the program
def millis():
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms

def unpack_cf_pwm(packed_pwm_data):
  unpacked_pwm_data = np.zeros((len(packed_pwm_data), 4))

  packed_pwm_data_int = np.zeros(packed_pwm_data.size, dtype=int)

  for i, l in enumerate(packed_pwm_data):
    packed_pwm_data_int[i] = int(l)

  for i, packed_pwm in enumerate(packed_pwm_data_int):
    #pwms = struct.upack('4H', packed_pwm);
    #print("m1,2,3,4: ", pwms)
    m1 = ( packed_pwm        & 0xFF) << 8
    m2 = ((packed_pwm >> 8)  & 0xFF) << 8
    m3 = ((packed_pwm >> 16) & 0xFF) << 8
    m4 = ((packed_pwm >> 24) & 0xFF) << 8
    unpacked_pwm_data[i][0] = m1
    unpacked_pwm_data[i][1] = m2
    unpacked_pwm_data[i][2] = m3
    unpacked_pwm_data[i][3] = m4

  return unpacked_pwm_data

def unpack_cf_imu(packed_imu_data_l, packed_imu_data_a):
  unpacked_imu_data = np.zeros((len(packed_imu_data_l), 6))

  mask = 0b1111111111

  packed_imu_data_l_int = np.zeros(packed_imu_data_l.size, dtype=int)
  packed_imu_data_a_int = np.zeros(packed_imu_data_a.size, dtype=int)

  for i, (l,a) in enumerate(zip(packed_imu_data_l, packed_imu_data_a)):
    packed_imu_data_l_int[i] = int(l)
    packed_imu_data_a_int[i] = int(a)

  for i, (packed_imu_l, packed_imu_a) in enumerate(zip(packed_imu_data_l_int, packed_imu_data_a_int)):

    lx = ( packed_imu_l        & mask)
    ly = ((packed_imu_l >> 10) & mask)
    lz = ((packed_imu_l >> 20) & mask)
    ax = ( packed_imu_a        & mask)
    ay = ((packed_imu_a >> 10) & mask)
    az = ((packed_imu_a >> 20) & mask)

    # scale back to normal (reflects values in custom cf firmware (sorry its so opaque!)
    lx = (lx / 10.23) - 50
    ly = (ly / 10.23) - 50
    lz = (lz / 10.23) - 50

    ax = (ax / 1.42)  - 360
    ay = (ay / 1.42)  - 360
    az = (az / 1.42)  - 360

    unpacked_imu_data[i][0] = lx
    unpacked_imu_data[i][1] = ly
    unpacked_imu_data[i][2] = lz
    unpacked_imu_data[i][3] = ax
    unpacked_imu_data[i][4] = ay
    unpacked_imu_data[i][5] = az

    #print("Unpacked IMU: ", unpacked_imu_data[i])

  return unpacked_imu_data
