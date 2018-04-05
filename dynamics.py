# File containing the base class dynamics

class Dynamics:
    # init class
    def __init__(self,dt, x_dim=12, u_dim = 4):
        self.timestep = dt
        self.x_dim = x_dim
        self.u_dim = u_dim

    # dimension check raises error if incorrect
    def _enforce_dimension(self, x, u):
        if np.size(x) != self.x_dim:
            raise ValueError('x dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(x)) + ', desired: ' + str(self.x_dim) )
        if np.size(u) != self.u_dim:
            raise ValueError('u dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.u_dim) )

def generate_data(dynam, sequence_len, num_iter):
    # generates a batch of data sequences for learning. Will be an array of (sequence_len x 2) sequences with state and inputs
