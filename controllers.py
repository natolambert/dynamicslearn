# File containing controllers for both collecting data and for on learned dynamics
import numpy as np
import cvxopt           # convex opt package

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

class randControl(Controller):
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

class MPControl(Controller):
    # MPC control, there will be two types
    # 1. random shooting control, with best reward being taken
    # 2. convext optimization solution on finite time horizon

    def __init__(self, dynamics_learned, dynamics_true, optimizer, N = 5, max_min= 'min', method = 'Shooter'):
        # initialize some variables

        dt = dynamics_true.get_dt
        dim = dynamics_true.get_dims[1]
        super().__init__(dt, dim=dim)

        self.dynamics_learned = dynamics_learned
        self.dynamics_true = dynamics_true

        # opt Parameters
        self.optimizer = optimizer      # function passed in to min or max
        self.max_min = 'min'            # min or max cost / reward fnc
        self.method = 'Shooter'         # default to random shooting MPC
        self.time_horiz = N             # time steps into future to look

    def control(self, current_state):
        # function that returns desired control output
        raise NotImplementedError('Not Yet Implemented')
