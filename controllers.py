# File containing controllers for both collecting data and for on learned dynamics
import numpy as np
import cvxopt           # convex opt package

# Import models files for MPC controller
from models import *

class Controller:
    # init class
    def __init__(self, dt, dim=4):
        self.dt = dt
        self.dim = dim
        self.var = [0]
        self.equil = np.zeros(dim)

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
    def __init__(self, dynamics, variance = .00001):
        # dt is update rate desired, more important for future subclasses
        # dim is the dimension of the control output
        # dynamics is an istance of the dynamics that provides info used for control
        # variance is the divergence from the equilibrium point
        dt = dynamics.get_dt
        dim = dynamics.get_dims[1]
        super().__init__(dt, dim=dim)

        # equilibrium point is from dynamics
        self.equil = dynamics.u_e
        self.var = variance

    def update(self):
        # returns a random control sample
        return self.equil + np.random.normal(scale=self.var,size=(self.dim))

    def setVar(self, newVar):
        self.var = newVar

class HoverPID(Controller):
    def __init__(self, dynamics, kP = 1, kI = 0, kD = 0, z_pt = 0):
        # dt is update rate desired, more important for future subclasses
        # dim is the dimension of the control output
        # dynamics is an istance of the dynamics that provides info used for control
        # acts by hovering around the z setpoint, and minimizing roll and pitch to 0
        dt = dynamics.get_dt
        dim = dynamics.get_dims[1]
        super().__init__(dt, dim=dim)

        # equilibrium point is from dynamics
        self.equil = dynamics.u_e

        # Contorl Parameters
        self.kP = kP
        self.kI = kI
        self.kD = kD

        # Setpoint
        self.z_pt = z_pt

    def update(self, state):
        # returns a random control sample
        z = state[2]
        pitch = state[4]
        self.error
        return self.equil + np.random.normal(scale=self.var,size=(self.dim))

    def setkP(self,kPnew):
        self.kP = kPnew

    def setkI(self,kInew):
        self.kI = kInew

    def setkD(self,kDnew):
        self.kD = kDnew

class PID:
    # Class for 1d PID controller
    def __init__(self, kP = 1, kI = 0, kD = 0, target = 0, integral_max = 100, integral_min = -100):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.target = target

        # terms for calculating PID values
        self.error = 0
        self.last_error = 0
        self.integral_error = 0
        self.integral_max =integral_max
        self.integral_min = integral_min


    def update(self, val):
        # updates the PID value for a given Value
        error = val - self.target
        self.error = error

        # Caps integral error
        self.integral_error = self.integral_error + error
        if self.integral_error > self.integral_max:
            self.integral_error = self.integral_max
        elif self.integral_error < self.integral_min:
            self.integral_error = self.integral_min

        # Calculate PID terms
        P_fact = self.kP*error
        I_fact = self.kI*(self.integral_error)
        D_fact = self.kD*(error-self.last_error)
        self.last_error = error

        return P_fact + I_fact + D_fact

    @property
    def kPID(self):
        # returns the PID values
        return [self.kP, self.kI, self.kD]

    @kPID.setter
    def kPID(self, kPIDnew):
        # sets new kPID values
        self.kP = kPIDnew[0]
        self.kI = kPIDnew[1]
        self.kD = kPIDnew[2]

# class PIDControl(Controller):

class MPController(Controller):
    # MPC control, there will be two types
    # 1. random shooting control, with best reward being taken
    # 2. convext optimization solution on finite time horizon

    def __init__(self, dynamics_learned, dynamics_true, Objective, N = 50, T=5, method = 'Shooter'):
        # initialize some variables
        # dynamics learned will be a model from models.py
        # dynamcis true is the dynamics file for getting some parameters
        # rewardORcost is of class Optimizer
        # N is number of random sequences to try
        # T is time horizon

        dt = dynamics_true.get_dt
        dim = dynamics_true.get_dims[1]
        super().__init__(dt, dim=dim)

        self.dynamics_model = dynamics_learned
        self.dynamics_true = dynamics_true

        # opt Parameters
        self.Objective = Objective   # function passed in to be min'd or max'd. Of class Objective
        self.method = 'Shooter'         # default to random shooting MPC
        self.time_horiz = T             # time steps into future to look
        self.N = N                      # number of samples to try when random

    def control(self, current_state):
        # function that returns desired control output

        if (self.method != 'Shooter'):
            raise NotImplementedError('Not Yet Implemented. Please use shooter random method')

        # Simulate a bunch of random actions and then need a way to evaluate reward
        N = self.N
        T = self.time_horiz
        # Makes controller to generate some action
        rand_controller = randController(self.dynamics_true)
        actions = [rand_controller.update() for i in range(N)]

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

        print('checking shapes')
        print(np.shape(X_sim))
        print(np.shape(actions_seq))

        # Evaluate all the sequences with the objective function, get index of best action

        # Load objective with simulated data
        self.Objective.loaddata(np.array(X_sim))

        # Calculate best actions
        mm_idx = self.Objective.compute_ARGmm()
        print(mm_idx)
        best_action = actions_seq[mm_idx]
        return best_action

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
        if (self.data == []):
            raise AttributeError('Data Not Loaded')
        print(np.shape(self.data))
        # Chooses data of sub-indices of each trajectory
        print(self.dim_to_eval)
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
            raise ValueError('Dimension of input does not match what was set')
