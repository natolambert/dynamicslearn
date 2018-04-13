# File containing controllers for both collecting data and for on learned dynamics
import numpy as np

class Controller:
    # init class
    def __init__(self, dt, dim=4):
        self.dt = dt
        self.dim = dim

    @property
    def getdims(self):
        return self.dt, self.dim

    # dimension check raises error if incorrect
    def _enforce_dimension(self, u):
        if np.size(u) != self.dim:
            raise ValueError('u dimension passed into controller does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.dim) )

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

    @property
    def get_var(self):
        return self.var

    @property
    def get_equil(self):
        return self.equil


# class PIDControl(Controller):

# class MPControl(Controller):
