# File containing controllers for both collecting data and for on learned dynamics
import numpy as np
# import cvxopt           # convex opt package

# Import models files for MPC controller
import time

class Controller:
    # Class controller is for forcing dimensions and certain properties.

    # init class
    def __init__(self, dt_update, dt_control, dim=4):
        self.dt_update = dt_update
        self.dt_control = dt_control
        # dt_control/dt_update must be an integer. This integer refers to the number of actions that the controller will take before an update. The controller update rate should be much slower than the dynamics update rate.

        self.dim = dim
        self.var = [0]

    # dimension check raises error if incorrect
    def _enforce_dimension(self, u):
        if np.size(u) != self.dim:
            raise ValueError('u dimension passed into controller does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.dim) )

    @property
    def get_dim(self):
        return self.dim

    @property
    def get_dt(self):
        return self.dt

    @property
    def get_var(self):
        return self.var

    @property
    def get_equil(self):
        return self.equil

class randController(Controller):
    def __init__(self, dynamics, dt_update, dt_control, variance = .00025):
        # dt is update rate desired, more important for future subclasses
        # dim is the dimension of the control output
        # dynamics is an istance of the dynamics that provides info used for control
        # variance is the divergence from the equilibrium point. The variance variable is important, because when it is too low it will not stray from equilibrium, but too high it will diverge rapidly
        dt_dynam = dynamics.get_dt
        dim = dynamics.get_dims[1]
        super().__init__(dt_update, dt_control, dim=dim)

        # equilibrium point is from dynamics
        self.equil = dynamics.u_e       # this is an input
        self.var = variance
        self.control = self.equil        # initialize equilibrium control

        # Index to track timing of controller / dynamics update
        self.i = 0
        self.Rdt = int(self.dt_control/self.dt_update)    # number of dynamics updates per new control update

    def update(self, _):
        # returns a random control sample around equilibrium
        # Create index to repeat control when below control update rates
        # ___ to take a start variable in general use, other contorllers use state info


        # print('Ratio is: ', Rdt, ' Index is: ', self.i)

        if ((self.i % self.Rdt) == 0):
            self.control = self.equil + np.random.normal(scale=self.var,size=(self.dim))

        # update index
        self.i += 1

        return self.control

class MPController(Controller):
    # MPC control, there will be two types
    # 1. random shooting control, with best reward being taken
    # 2. convext optimization solution on finite time horizon

    def __init__(self, dynamics_learned, dynamics_true, dt_update, dt_control, Objective, N = 100, T=10, variance = .00001, method = 'Shooter'):
        # initialize some variables
        # dynamics learned will be a model from models.py
        # dynamcis true is the dynamics file for getting some parameters
        # rewardORcost is of class Optimizer
        # N is number of random sequences to try
        # T is time horizon

        # time step to be used inthe future when control update rate != dynamics update rate
        self.dt_dynam = dynamics_true.get_dt
        dim = dynamics_true.get_dims[1]
        super().__init__(dt_update, dt_control, dim=dim)

        self.dynamics_model = dynamics_learned
        self.dynamics_true = dynamics_true

        # time variance of control variable intialization
        self.i = 0
        self.control = dynamics_true.u_e

        # opt Parameters
        self.Objective = Objective   # function passed in to be min'd or max'd. Of class Objective
        self.method = 'Shooter'         # default to random shooting MPC
        self.time_horiz = T             # time steps into future to look
        self.N = N                      # number of samples to try when random
        self.var = variance

        self.rand_controller = randController(self.dynamics_true, self.dt_control, self.dt_control, variance = self.var)

        self.zeros = np.zeros(12)


    def update(self, current_state):
        # function that returns desired control output

        if (self.method != 'Shooter'):
            raise NotImplementedError('Not Yet Implemented. Please use shooter random method')

        # initialize some variables for only updating control at correct
        # if dt_u == dt_x R = 1, so always updates
        Rdt = int(self.dt_control/self.dt_update)    # number of dynamics updates per new control update

        # Simulate a bunch of random actions and then need a way to evaluate reward
        N = self.N
        T = self.time_horiz

        if ((self.i % Rdt) == 0):
            start_time = time.time()
            # Makes controller to generate some action
            # passes the dt_control twice so that every call of .update() generates a unique random action. When dimulating a sequence at a specific rate, the randController has a built in ticker that tracks whether this dynamics update it should give a new control or not.
            actions = [self.rand_controller.update(self.zeros) for i in range(N)]

            # Extends control to the time horizon defined in init
            actions_list = []
            for i in range(T):
                actions_list.append(actions)    #creates actions depth wise here
            actions_seq = np.dstack(actions_list)
            actions_seq = np.swapaxes(actions_seq,1,2)        # Keeps tuple in the form (n, traj_idx, state/input_idx)

            # simulates actions on learned dynamics, need to fill in matrix
            X_sim = [] # np.tile(current_state,(N,1))  # tiled matrix of x0 = current_state
            for n in range(N):
                seq_sim = simulate_learned(self.dynamics_model, actions_seq[n,:,:], x0=current_state)
                # append sequence to array
                X_sim.append(seq_sim)

            # print('checking shapes')
            # print(np.shape(X_sim))
            # print(np.shape(actions_seq))

            # Evaluate all the sequences with the objective function, get index of best action

            # Load objective with simulated data
            self.Objective.loaddata(np.array(X_sim))

            # Calculate best actions
            mm_idx = self.Objective.compute_ARGmm()
            # print(mm_idx)
            best_action = actions_seq[mm_idx]
            self.control = best_action[0]
            # print(self.control)

            # print(time.time()-start_time)

        self.i += 1

        return self.control

class Objective():
    # class of objective functions to be used in MPC and maybe future implementations. Takes in sets of sequences when optimizing!
    def __init__(self, function, maxORmin, dim, dim_to_eval=[], data =[],  choose_dim = True):

        # lambda or other function to max or min based on state and or input
        self.optimizer = function

        # Dimensions of the data to evaluate objective on
        self.dim = dim              # dimension of each vector that will be eval
        self.data = data
        self.choose_dim = choose_dim

        if (choose_dim == False):
            self.dim_to_eval = range(dim[1])
        else:
            self.dim_to_eval = dim_to_eval

        # sets max and argmax etc
        self.maxORmin = maxORmin
        if (maxORmin == 'max'):
            self.mm = np.max              # mm is my notation for maxormin
            self.argmm = np.argmax
        elif (maxORmin == 'min'):
            self.mm = np.min
            self.argmm = np.argmin
        else:
            raise ValueError('Pass useable optimization function max or min')

    def loaddata(self,data):
        # loads data to be optimized over
        self.data = data

    def compute_mm(self):
        # computes the min or max of the objective function given data
        if (self.data == []):
            raise AttributeError('Data Not Loaded')

        # Chooses data of sub-indices of each trajectory
        data_eval = self.data[:,:,self.dim_to_eval]


        objective_vals = [np.sum(self.eval(traj),axis=0) for traj in data_eval]
        # print(np.shape(objective_vals))
        mm = self.mm(objective_vals)

        return mm

    def compute_ARGmm(self):
        # computes the ARGmax or min over the data
        # if (self.data == []):
        #     raise AttributeError('Data Not Loaded')
        # print(np.shape(self.data))
        # Chooses data of sub-indices of each trajectory
        # print(self.dim_to_eval)
        data_eval = self.data[:,:,self.dim_to_eval]
        objective_vals = [np.sum(self.eval(traj),axis=0) for traj in data_eval]
        # print(np.shape(objective_vals))
        mm_idx = self.argmm(objective_vals)

        return mm_idx

    def eval(self, input_array):
        # takes in an inpute array and returns the objective Value for each row
        # Checks to make sure dimension is right
        self._enforce_dimension(input_array[0,:])
        val = [np.sum(self.optimizer(vect)) for vect in input_array ]
        return val

    def _enforce_dimension(self, input_vector):
        if (self.dim != np.size(input_vector)):
            #print(np.shape(input_vector),np.shape(self.dim))
            raise ValueError('Dimension of input does not match what was set')
