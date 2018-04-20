# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScaler

def stack_pairs(states, actions):
    # returns a list of 2-tuples of state vectors with action vectors
    # works with change states as well
    dim_states = np.shape(states)
    dim_actions = np.shape(actions)
    if (dim_states[0] != dim_actions[0]):
        raise ValueError('states and actions not same length')

    lst = []
    for (a,s) in zip(states,actions):
        lst.append([a,s])

    return np.array(lst)

def stack_trios(change_states, states, actions):
    # returns a list of 3-tuples of change in state, state vectors, and action vectors
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
    # takes in a array of states, and reurns a array of states that is the     #   difference between the current state and next state.
    # NOTE: Does not return a value for the first point given
    # Above is bevause we are trying to model x_t+1 = f(x_t,a_t)

    dim = np.shape(states)
    delta = np.zeros((dim[0]-1,dim[1]))

    for (i,s) in enumerate(states[1:,:]):
        delta[i-1,:] = states[i,:]-states[i-1,:]

    return delta

def normalize_states(delta_states, ScaleType = StandardScaler, x_dim = 12):
    # normalizes states to standard scalars
    # delta states should be a n by x_dim array
    # NOTE: I chose to implement this function because there may be a time when
    #   we want to scale different states differently, and this is the place to do it

    if (x_dim != 12):
        raise ValueError('normalize_states not designed for this state vector')
    scaler = ScaleType()
    scaled = scaler.fit(delta_states)
    return scaled

def sequencesXU2array(X, U, normalize = False):
    # Uses other functions to take in two arrays X,U that are 3d arrays of sequences of states and actions
    # n = num sequences
    # l = len(sequences)
    # dimx, dimu easy
    if normalize:
        raise NotImplementedError('Have not implemented normalization')

    n, l, dimx = np.shape(X)
    _, _, dimu = np.shape(U)

    # # pre-allocates matrix to store all 3-tuples, n-1 because need delta state
    # data = np.zeros(((n-1)*l,dimx+dimu))
    seqs = []
    for (seqX, seqU) in zip(X,U):
        # generates the changes in states from raw data
        delta_states = states2delta(seqX)

        # generates tuples of length l-1, with the 3 elements being
        # dx : change in state vector from time t
        # x : state vector at time t
        # u : input vector at time t
        dx_x_u_t = stack_trios(delta_states,seqX[:-1,:], seqU[:-1,:])
        seqs.append(dx_x_u_t)

    # reshape data into a long list of dx, x, u pairs for training
    data = np.reshape(seqs, (n*(l-1),3))
    return data

def l2array(list_arrays):
    # 2-tuple list to array. Needed for all the trios of data that return columns where each column is a list of states and not a 2d array of states etc. This may be slightly poor practice.
    arr = []
    for l in list_arrays:
        arr.append(l)
    return np.array(arr)
