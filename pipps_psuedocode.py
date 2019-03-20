import numpy as np
from model_general_nn import GeneralNN
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from collections import OrderedDict
# For training policies following the Probablistic Inference for PArticle-Based Policy Search Guidelines
# http://proceedings.mlr.press/v80/parmas18a.html

# Need to load: a dynamics model, a policy to be updated on 

# For visualizing the computation graph
from torchviz import make_dot

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
        self.activation = nn.ReLU()
        self.d = nn_params['dropout']

        # create object nicely
        layers = []
        layers.append(('pipps_input_lin', nn.Linear(self.n_in, self.hidden_w)))       # input layer
        layers.append(('pipps_input_act', self.activation))
        # layers.append(nn.Dropout(p=self.d))
        for d in range(self.depth):
            # add modules
            # input layer
            layers.append(
                ('pipps_lin_'+str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('pipps_act_'+str(d), self.activation))
            # layers.append(nn.Dropout(p=self.d))

        # output layer
        layers.append(('pipps_out_lin', nn.Linear(self.hidden_w, self.n_out,bias=True)))
        # print(*layers)
        self.features = nn.Sequential(OrderedDict([*layers]))

        # create scalars, these may help with probability predictions
        # note these are not tensors, so lead to difficulty in propogating gradients
        self.scalarX = StandardScaler()
        self.scalarU = MinMaxScaler(feature_range=(-1, 1))
        # store the information manually, init empty
        self.state_means = []
        self.state_vars = []

        self.dynam_model = dynam_model

        # sets policy parameters
        self.T = policy_update_params['T']
        self.P = policy_update_params['P']
        self.lr = policy_update_params['learning_rate']
        
    
    def init_weights_orth(self):
        # inits the NN with orthogonal weights
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight, gain =1)
                # nn.init.uniform_(m.bias, -.5, 0)
                # nn.init.normal_(m.weight, mean=0,std=.1)

        self.features.apply(init_weights)

    def forward(self, x, normalize = False):
        """
        Standard forward function necessary if extending nn.Module.
        """
        if normalize:
            x = (x - self.state_means) / torch.sqrt(self.state_vars)

        x = self.features(x)
        return x

    def get_action(self, X):
        """
        Gets an action from the NN. Scaled based on the action distribution at previous update
        """
        dx = len(X)

        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        # normU = self.scalarU.transform(U.reshape(1, -1))

        # input = torch.Tensor(np.concatenate((normX, normU), axis=1))
        input = torch.Tensor(normX)

        NNout = self.forward(input).data[0]

        # If probablistic only takes the first half of the outputs for predictions
        NNout = self.postprocess(NNout).ravel()

        return NNout

    def scale_actions(self, actions):
        """
        Scales outputs to be from action min to action max
        """

    def postprocess(self, U):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
        """
        # de-normalize so to say
        U = self.scalarU.inverse_transform(U.reshape(1, -1))
        U = U.ravel()
        return np.array(U)


    def set_statespace_normal(self, means, variances):
        """
        Takes in a series of means and variances for the logged data of the state space
        - Need this so the different states can be normalized to unit normal before passing into net
        """
        if means == [] and variances == []:
            print("loading data distribution from the dynamics model")
            self.state_means = self.dynam_model.scalarX_tensors_mean
            self.state_vars = self.dynam_model.scalarX_tensors_var
        else:
            self.state_means = torch.Tensor(means)
            self.state_vars = torch.Tensor(variances)


    def normalize_state(self,x):
        """
        NOTE: The dynamics model handles this
        takes in a a state x and normalizes to unit normal to pass into neural net
        - This needs to be used for the forward call
        """
        raise NotImplementedError("This is done in model_general_nn")

    def gen_policy_rollout(self, observations, dynam_model):
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
            norm_dist = torch.distributions.Normal(0, 1)

            # for storing the costs and gradients
            costs = torch.Tensor()
            baselines = torch.Tensor()
            log_probabilities = torch.Tensor()
            states = torch.Tensor()
            ''' 
            # for all of these values, think a row as an updating time series for each particle
            costs = torch.zeros((self.P, self.T))
            baselines = torch.zeros((self.P, self.T))
            probabilities = torch.zeros((self.P, self.T))

            # This is what we would do for the states, but it's more efficient to concatenate them
            states = torch.Tensor((self.P,self.T, self.n_in))
            '''
            
            # Change in state vs raw state key
            pred_key = dynam_model.change_state_list
            # print(pred_key)

            # TODO: Generate initial states for each particle based on the distribution of data it was trained on
            bound_up = torch.Tensor(np.max(observations,0))
            bound_low = torch.Tensor(np.min(observations, 0))

            obs_dist = torch.distributions.uniform.Uniform(bound_low,bound_up)

            # iterate through each particle for the states and probabilites for gradient
            for p in range(self.P):

                # Choose the dynamics model from the ensemble
                num_ens = dynam_model.E
                if num_ens == 0:
                    model = dynam_model
                else:
                    model_idx = random.randint(0, num_ens-1)
                    model = dynam_model.networks[model_idx]
                
                num_obs = np.shape(observations)[1]
                # x0 = torch.Tensor(observations[random.randint(0,num_obs),:])
                x0 = obs_dist.sample()

                # TODO: Normalize the states before passing into the NN

                state_mat = x0.view((1, 1, -1))
                # print(state_mat)
                # state_mat = x0.unsqueeze(0).unsqueeze(0)    # takes a (n_in) vector to a (1,1,n_in) Tensor
                # torch.cat((), axis = 1) to cat the times
                # torch.cat((), axis = 0) to cat the particle
                # is there a way to do this without unsqueeze? Seems like the most efficient way
                log_prob_vect = torch.Tensor([1])   #states the column with 1 for concat'ing
    
                for t in range(self.T):
                    # generate action from the policy
                    action = self.forward(state_mat[0,t,:])
                    # print(action)
                    # quit()

                    # forward pass current state to generate distribution values from dynamics model
                    means, var = model.distribution(state_mat[0, t, :], action)

                    # sample the next state from the means and variances of the state transition probabilities
                    vals = var*norm_dist.sample((1, self.n_in)) + means
                    # need to account for the fact that some states are change in and some are raw here


                    # batch mode prob calc
                    # log_probs = -.5*torch.abs(vals - means)/var
                    log_probs = -.5*torch.abs(vals - means)/(var**2)

                    # for s in range(self.n_in):
                    #     # sample predicted new state for each element
                    #     val = var[s]*np.random.normal()+means[0]    # sample from the scaled gaussian with y = sigma*x + mu

                    #     # calculate probability of this state for each sub state
                    #     p = -.5*(val-means[0])/var[0]
                    # states = torch.cat((states, state), 0)
                    # probabilities = torch.cat((probabilities, p), 0)

                    # reduce the probabilities vector to get a single probability of the state transition
                    log_prob = torch.sum((log_probs), 1)
                    log_prob_vect = torch.cat((log_prob_vect, log_prob))

                    state = torch.Tensor(vals).view((1, 1, -1))

                    state_mat = torch.cat((state_mat, state), 1) # appends the currnt state to the current particle, without overwriting the otherone
                
                # print(state_mat)
                # calculates costs
                # idea ~ calculate the cost of each each element and then do an cumulative sum for the costs
                # use torch.cumsum
                c_list = []
                for state in state_mat.squeeze():
                    c_row = self.cost_fnc(state)
                    c_list.append(c_row)
                c_list = torch.stack(c_list)
                
                # note we calc the cum sum on the flipped tensor, then flip costs back
                cost_cum = torch.cumsum(torch.flip(c_list,[0]),0)
                # Assembles the arrays for the current particle
                
                
                # costs were given above
                costs = torch.cat((costs, torch.flip(cost_cum, [0]).view(1,-1)), 0)

                # update the states array for each particle
                states = torch.cat((states, state_mat), 0) 

                # concatenates the vector of prob at each time to the 2d array
                log_probabilities = torch.cat(
                    (log_probabilities, log_prob_vect.view((1, -1))), 0)



            # calculates baselines as the leave one out mean for each particle at each time
            costs_summed = torch.sum(costs,0)
            costs_summed_exp = costs_summed.expand_as(costs)
            costs_leave_one_out = costs_summed_exp-costs
            baselines = costs_leave_one_out/(self.P-1)
            

            # freezes gradients on costs and baselines
            # these two lines of code actually do nothing, but are for clarification
            # costs.requires_grad_(requires_grad=False)
            # baselines.requires_grad_(requires_grad=False)
            # . detach() is another way to ensure this
            """
            RuntimeError: you can only change requires_grad flags of leaf variables. If you want to use a 
               computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().
            """
            costs_d = costs.detach()
            baselines_d = baselines.detach()
            # costs_d = costs.requires_grad_(False)
            # baselines_d = baselines.requires_grad_(False)
            # print(baselines)
            # print(probabilities)
            # print(costs)
            print(log_probabilities)
            return states, log_probabilities, costs_d, baselines_d

    def policy_step(self, observations):

        optimizer = torch.optim.Adam(super(PIPPS_policy, self).parameters(), lr=self.lr)

        # raise NotImplementedError("Core method not implemented yet")
        optimizer.zero_grad()
        # simulate trajectories through the dynamics model with the policy

        # generate the probability of states given action, with the gradients connected to the NN object

        # Set up the weighted mean computation
        states, probabilities, costs, baselines = self.gen_policy_rollout(observations, self.dynam_model)

        # turn off the gradients on the mean cost and the baseline

        # Calculate the gradient (the expectation in the paper)
        # the values are of the form (#P, T), so we first recude across dim 1 to get (#P,1) and then the mean to get the value
        # take the mean over each particle for the sum of the trajectories
        weighted_costs = torch.mean(torch.sum((probabilities*(costs-baselines)), dim=1), dim=0)
        print("weighted costs: ", weighted_costs)
        # print(self.features[0].weight)
        
        # dot = make_dot(weighted_costs) #, params=dict(self.features.named_parameters()))
        # # dot = make_dot(weighted_costs, params=dict(
        #     # self.features.named_parameters()))

        # dot.view()
        # call gradient.step based on a policy update parameter
        weighted_costs.backward()
        # print(self.features[4].weight.grad)
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

    def viz_comp_graph(self, var):
        # inputs = torch.randn(self.n_in)
        # y = f(Variable(inputs), self.features)
        # make_dot(y, self.features)
        dot = make_dot(var, params=dict(self.features.named_parameters()))
        dot.view()
        quit()



