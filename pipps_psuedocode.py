import numpy as np
from model_general_nn import GeneralNN
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For training policies following the Probablistic Inference for PArticle-Based Policy Search Guidelines
# http://proceedings.mlr.press/v80/parmas18a.html

# Need to load: a dynamics model, a policy to be updated on 


'''
 Some variable names and intuition:
- G_h = sum_{t=h}^T c(x), is the sum of future cost. c(x) is cost at state x
- 

'''

# The code we need is the code that generates a policy in a general fashion and can generate probabilities from
'''
Directly from the PIPPs paper, a model-based gradeint derives:
E[SUM_t (d/dtheta log p(x | x; theta)) (G(x) - b)]
In this case, 
    G_t is the sum of the cost up to time t, (make sure gradient on this tensor is off)
    b_t is the baseline at a given time, can be the sample mean of the particles at time t
        consider the leave one out baseline where the baseline at a given particle i 
        is the average of all the other particles (in terms of cost)
    The probability of the transition from one state to another will be interesting to compute
        This is just the state transition probability (can we use the dynamics model we have as a prob?)
    

'''


class prob_dynam_model(GeneralNN):
    """
    Returns deterministic probabilities for any given state, conditioned on a action OR a a policy
    """
    def __init__(self):
        print("TODO - not sure if this or two functions is the best way to do this")

class PIPPS_policy(nn.Module):

    def __init__(self, nn_params, policy_update_params):
        # Takes in the same parameters dict as the dynamics model for my convenience
        super(PIPPS_policy, self).__init__()
        
        # sets a bunch of NN parameters
        self.prob = nn_params['bayesian_flag']  # in this case it makes the output probablistc
        self.hidden_w = nn_params['hid_width']
        self.depth = nn_params['hid_depth']
        self.n_in = nn_params['dx']
        self.n_out = nn_params['du']
        self.activation = nn_params['activation']
        self.d = nn_params['dropout']

        # create object nicely
        layers = []
        layers.append(nn.Linear(self.n_in, self.hidden_w)
                      )       # input layer
        layers.append(self.activation)
        layers.append(nn.Dropout(p=self.d))
        for d in range(self.depth):
            # add modules
            # input layer
            layers.append(nn.Linear(self.hidden_w, self.hidden_w))
            layers.append(self.activation)
            layers.append(nn.Dropout(p=self.d))

        # output layer
        layers.append(nn.Linear(self.hidden_w, self.n_out))
        self.features = nn.Sequential(*layers)

        # create scalars, these may help with probability predictions
        self.scalarX = StandardScaler()
        self.scalarU = MinMaxScaler(feature_range=(-1, 1))

        # sets policy parameters
        self.N = policy_update_params['N']
        self.T = policy_update_params['T']
        self.lr = policy_update_params['learning_rates']
        
    
    def init_weights_orth(self):
        # inits the NN with orthogonal weights
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight)

        self.features.apply(init_weights)

    def forward(self, x):
        """
        Standard forward function necessary if extending nn.Module.
        """

        x = self.features(x)

    def predict(self, X, U):
        """
        Given a state X and input U, predict the change in state dX. This function is used when simulating, so it does all pre and post processing for the neural net
        """
        dx = len(X)

        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))

        input = torch.Tensor(np.concatenate((normX, normU), axis=1))

        NNout = self.forward(input).data[0]

        # If probablistic only takes the first half of the outputs for predictions
        NNout = self.postprocess(NNout).ravel()

        return NNout

    def postprocess(self, U):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
        """
        # de-normalize so to say
        U = self.scalarU.inverse_transform(U.reshape(1, -1))
        U = U.ravel()
        return np.array(U)

    def policy_step(self):
        def gen_policy_rollout(x0, self):
            # initialize statespace array in pytorch tensor -> need that good good gradient!
            raise NotImplementedError("Need to implement the rollouts with tensors, or maybe not for gradients")

        raise NotImplementedError("Core method not implemented yet")
        # simulate trajectories through the dynamics model with the policy

        # generate the probability of states given action, with the gradients connected to the NN object

        # Set up the weighted mean computation

        # turn off the gradients on the mean cost and the baseline

        # Calculate the gradient (the expectation in the paper)

        # call gradient.step based on a policy update parameter

        # log prob term psuedo
        # -0.5 * \sum_{state dimensions}((particle - mu(x, policy(x, theta)) / sigma(x, policy(x, theta))) ^ 2


    
    def set_baseline_function(baseline, baseline_explanation = ''):
        self.baseline_fnc = baseline
        self.baseline = 0

    def set_cost_function(cost, cost_explanation = ''):
        self.cost_fnc = cost

