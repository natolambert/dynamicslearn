# file for data utilities
import numpy as np
from sklearn.preprocessing import StandardScalar

def stack_pairs(states, actions):
    # returns a list of 2-tuples of state vectors with action vectors
    # works with change states as well
    dim_states = np.shape(states)
    dim_actions = np.shape(actions)
    if (dim_states[0] != dim_action[0]):
        raise ValueError('states and actions not same length')

    lst = []
    for (a,s) in zip(states,actions):
        lst.append([a,s])

    return lst

def states_to_delta(states):
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
