# File containing controllers for both collecting data and for on learned dynamics

class Controller:
    # init class
    def __init__(self, dt, dim=4):
        self.dt = dt
        self.dim = dim

    @property
    def dims(self):
        return self.dt, self.dim

    # dimension check raises error if incorrect
    def _enforce_dimension(self, u):
        if np.size(u) != self.dim:
            raise ValueError('u dimension passed into controller does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.dim) )

class random(Controller):
    def __init__(self, dt, dim, dynamics, variance = .001):
        # dt is update rate desired, more important for future subclasses
        # dim is the dimension of the control output
        # dynamics is an istance of the dynamics that provides info used for control
        # variance is the divergence from the equilibrium point
        super().__init__(dt, dim=dim)

        # equilibrium point is from dynamics
        self.equil = dynamics.u_e
        self.var = variance

    def update(self):
        # returns a random control sample
        return self.equil + np.random.normal(scale=self.var,size=(self.dim))

    @property
    def var(self):
        return self.var

    @property
    def equil(self):
        return self.equil


# class PID(Controller):

# class MPC(Controller):
