import numpy as np
from model_general_nn import GeneralNN
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
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

    def __init__(self, nn_params, policy_update_params, dynam_model):
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
        layers.append(nn.Linear(self.n_in, self.hidden_w))       # input layer
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

        self.dynam_model = dynam_model

        # sets policy parameters
        self.N = policy_update_params['N']
        self.T = policy_update_params['T']
        self.P = 10
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
        return x

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

    def policy_step(self, x0):

        def gen_policy_rollout(x0, dynam_model):
            """
            Does the majority of the work in Pytorch for generating the policy gradient for PIPPS
             - input: a current state and a dynam_model to generate data and gradients with
             - output: 
                - a list of states evolution over time for each particle
                - a list of probabilities conditioned on the policy at each time for each particle
                - a list of costs over time for each particle
                - a list of baselines for each particle (ideal is unbiased leave one out estimator)
            """
            # initialize statespace array in pytorch tensor -> need that good good gradient!
            # raise NotImplementedError("Need to implement the rollouts with tensors, or maybe not for gradients")

            # for sampling the state progression
            norm_dist = torch.distributions.Normal(0,1)

            # for storing the costs and gradients 
            # for all of these values, think a row as an updating time series for each particle
            costs = torch.zeros((self.P, self.T))
            baselines = torch.zeros((self.P, self.T))
            probabilities = torch.zeros((self.P, self.T))

            # iterate through each particle for the states and probabilites for gradient
            for p in self.P:

                # Choose the dynamics model from the ensemble 
                num_ens = dynam_model.E
                if self.E == 0: model = dynam_model
                else:
                    model_idx = random.randint(0,num_ens)
                    model = dynam_model.networks[model_idx]

                state = torch.Tensor(x0)
                for t in range(self.T):
                    # generate action from the policy
                    action = self.forward(state)
                    # TODO: to properly backprop through the policy, cannot overwite state, we need to store state in a big array

                    # forward pass current state to generate distribution values from dynamics model
                    means, var = model.distribution(state, action)

                    # sample the next state from the means and variances of the state transition probabilities
                    vals = var*norm_dist.sample((1,self.n_in)) + means
                    
                    # batch mode prob calc
                    probs = -.5*(vals - means)/var

                    # for s in range(self.n_in):
                    #     # sample predicted new state for each element
                    #     val = var[s]*np.random.normal()+means[0]    # sample from the scaled gaussian with y = sigma*x + mu

                    #     # calculate probability of this state for each sub state
                    #     p = -.5*(val-means[0])/var[0]
                    states = torch.cat((states, state),0)
                    probabilities = torch.cat((probabilities, p),0)

                    # reduce the probabilities vector to get a single probability of the state transition 
                    prob = torch.prob(probabilities, axis =0)

                    state = torch.Tensor(vals)


                # calculates costs
                # idea ~ alculate the cost of each each element and then do an cumulative sum for the costs
                # use torch.cumsum
                for t in range(self.T):
                    c_row = self.cost_fnc(states[t,:])

            # calculates baselines as the leave one out mean for each particle at each time
            
            # freezes gradients on costs and baselines
            # these two lines of code actually do nothing, but are for clarification 
            costs.requires_grad_(requires_grad=False)
            baselines.requires_grad_(requires_grad=False)
            # . detach() is another way to ensure this
            # costs.detach()
            # baselines.detach()
            
            return states, probabilities, costs, baselines

        optimizer = torch.optim.Adam(super(PIPPS_policy, self).parameters(), lr=self.lr)

        # raise NotImplementedError("Core method not implemented yet")
        optimizer.zero_grad()
        # simulate trajectories through the dynamics model with the policy

        # generate the probability of states given action, with the gradients connected to the NN object

        # Set up the weighted mean computation
        states, probabilities, costs, baselines = gen_policy_rollout(x0, self.dynam_model)

        # turn off the gradients on the mean cost and the baseline

        # Calculate the gradient (the expectation in the paper)
        # the values are of the form (#P, T), so we first recude across dim 1 to get (#P,1) and then the mean to get the value
        loss = torch.mean(torch.sum((probabilities*(costs-baselines)), dim=1), dim=0)

        # call gradient.step based on a policy update parameter
        loss.backwards()
        optimizer.step()

        # log prob term psuedo
        # -0.5 * \sum_{state dimensions}((particle - mu(x, policy(x, theta)) / sigma(x, policy(x, theta))) ^ 2

        

        '''
        To sum all elements of a tensor:

        torch.sum(outputs)  # gives back a scalar
        To sum over all rows(i.e. for each column):

        torch.sum(outputs, dim=0)  # size = [1, ncol]
        To sum over all columns(i.e. for each row):

        torch.sum(outputs, dim=1)  # size = [nrow, 1]
        '''
    
    def set_baseline_function(self, baseline, baseline_explanation = ''):
        self.baseline_fnc = baseline
        self.baseline = 0

    def set_cost_function(self, cost, cost_explanation = ''):
        self.cost_fnc = cost



