
# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datetime import timedelta
import struct
import os
import matplotlib.pyplot as plt

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
        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), flight_len#np.array(Time)

def stack_pairs(states, actions):
    '''
    Returns a list of 2-tuples of state vectors with action vectors
    works with change states as well.  inputs are [(n by dx), (n by du)] and the output is [(n)] where each element is an array of an state and action
    '''
    dim_states = np.shape(states)
    dim_actions = np.shape(actions)
    if (dim_states[0] != dim_actions[0]):
        raise ValueError('states and actions not same length')

    lst = []
    for (a,s) in zip(states,actions):
        lst.append([a,s])

    return np.array(lst)

def stack_trios(change_states, states, actions):
    '''
    Returns a list of 3-tuples of change in state, state vectors, and action vectors. inputs are of shape [(n by dx), (n by dx), (n by du)] and the output is an array of arrays of shape [(n)] each element is a 1x3 arrary of states etc
    '''
    dim_change = np.shape(change_states)
    dim_states = np.shape(states)
    dim_actions = np.shape(actions)
    if (dim_states != dim_change):
        raise ValueError('states and change state not same dim')

    if (dim_states[0] != dim_actions[0]):
        raise ValueError('states and actions not same length')

    lst = []
    for (d,a,s) in zip(change_states,states,actions):
        lst.append([d,a,s])
    # print('das shape: ', np.array(lst).shape)
    return np.array(lst)

def states2delta(states):
    '''
    Takes in a array of states, and reurns a array of states that is the    difference between the current state and next state.
    NOTE: Does not return a value for the first point given
    Above is bevause we are trying to model x_t+1 = f(x_t,a_t). Input is of shape [n by dx], output is [n-1 by dx]
    '''

    dim = np.shape(states)
    delta = np.zeros((dim[0]-1,dim[1]))
    for (i,s) in enumerate(states[1:,:]):
        delta[i-1,:] = states[i,:]-states[i-1,:]
    return delta

def normalize_states(delta_states, ScaleType = StandardScaler, x_dim = 12):
    '''
    normalizes states to standard scalars
    delta states should be a n by x_dim array
    NOTE: I chose to implement this function because there may be a time when
      we want to scale different states differently, and this is the place to do it. Could be used for least squares, NN has its own implementation
      '''

    if (x_dim != 12):
        raise ValueError('normalize_states not designed for this state vector')
    scaler = ScaleType()
    scaled = scaler.fit(delta_states)
    return scaled

def sequencesXU2array(Seqs_X, Seqs_U, normalize = False):
    '''
    Uses other functions to take in two arrays X,U that are 3d arrays of sequences of states and actions (nseqs, idx_sed, idx_state). Returns an array of trios for training of size (nseqs*(l-1), 3) the 3 corresponds to arrays of change state, curstate, input arrays
    n = num sequences
    l = len(sequences)
    dimx, dimu easy
    '''
    if normalize:
        raise NotImplementedError('Have not implemented normalization')

    n, l, dimx = np.shape(Seqs_X)
    _, _, dimu = np.shape(Seqs_U)

    # # pre-allocates matrix to store all 3-tuples, n-1 because need delta state
    # data = np.zeros(((n-1)*l,dimx+dimu))
    seqs = []
    Seqs_dX = Seqs_X[:,1:,:]-Seqs_X[:,:-1,:]
    print(np.shape(Seqs_dX))
    print(np.shape(Seqs_X))
    print(np.shape(Seqs_U))

    for (seqdX, seqX, seqU) in zip(Seqs_dX, Seqs_X,Seqs_U):
        # generates the changes in states from raw data
        delta_states = states2delta(seqX)

        # generates tuples of length l-1, with the 3 elements being
        # dx : change in state vector from time t
        # x : state vector at time t
        # u : input vector at time t
        dx_x_u_t = stack_trios(seqdX[:,:], seqX[:-1,:], seqU[:-1,:])
        seqs.append(dx_x_u_t)

    print(np.shape(seqs))
    # reshape data into a long list of dx, x, u pairs for training

    if len(np.shape(seqs))  == 4:
      data = np.reshape(seqs, (n*(l-1),3, dimx))
    else:
      data = np.reshape(seqs, (n*(l-1),3))
    return data

def l2array(list_arrays):
    '''
    2-tuple list to array. Needed for all the trios of data that return columns where each column is a list of states and not a 2d array of states etc. This may be slightly poor practice. Use this to convert one of the above functions returns an (nx3) array of arrays. For example, call l2array(data[:,0]) to generate a 2d numpy array of the next state data of generated sequences.
    '''
    arr = []
    for l in list_arrays:
        arr.append(l)
    return np.array(arr)

def loadcsv(filename):
    '''
    Loads a csv / txt file with comma delimiters to be used for the real data system into this Setup
    format in csv: accel x, accel y, accel z, roll, pitch, yaw, pwm1, pwm2, pwm3, pwm4 timestamp

    Data has -1 in PWM columns for invalid bits / bits not to train on
    '''
    data = np.genfromtxt(filename, delimiter=',', invalid_raise = False)[:,0:11]
    data = data[~np.isnan(data).any(axis=1)]
    data = data[data[:,7]!=-1,:]
    states = data[:,0:6]
    actions = data[:,6:10]
    return states,actions

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
