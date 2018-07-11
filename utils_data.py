# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    # print(np.shape(Seqs_dX))
    # print(np.shape(Seqs_X))
    # print(np.shape(Seqs_U))

    for (seqdX, seqX, seqU) in zip(Seqs_dX, Seqs_X,Seqs_U):
        # generates the changes in states from raw data
        delta_states = states2delta(seqX)

        # generates tuples of length l-1, with the 3 elements being
        # dx : change in state vector from time t
        # x : state vector at time t
        # u : input vector at time t
        dx_x_u_t = stack_trios(seqdX[:,:], seqX[:-1,:], seqU[:-1,:])
        seqs.append(dx_x_u_t)

    # reshape data into a long list of dx, x, u pairs for training
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
