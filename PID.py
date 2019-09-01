'''PID class to help implement PID policy'''

import numpy as np

class PID():
    def __init__(self, desired,
                    kp, ki, kd,
                    ilimit, dt, outlimit = np.inf,
                    samplingRate = 0, cutoffFreq = -1,
                    enableDFilter = False, memory = 5):

        self.error = 0 #proportional error
        self.error_prev = [0 for i in range(memory)] #keeps track of previous errors length memory
        self.integral = 0 #integral error
        self.deriv = 0 #derivative error
        self.out = 0 #output of PID
        self.bias = 1e-5
        self.memory = memory

        self.desired = desired #goal parameter to base errors off of

        #PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.ilimit = ilimit  #Integral error limit
        self.outlimit = outlimit #Output error limit

        # time steps for changing step size of PID response
        self.dt = dt

    def update(self, measured):
        self.out = 0 #reset the output

        self.error_prev = self.error_prev[1:] + [self.error] #update the previous error

        #proportional error
        self.error = measured - self.desired
        self.out += self.kp*self.error

        #derivative error
        self.deriv = (self.error-self.error_prev[self.memory - 1]) / self.dt
        self.out += self.deriv*self.kd

        #integral error
        self.integral = np.sum(self.error_prev)*self.dt
        # limit the integral term
        if self.ilimit !=0:
            self.integral = np.clip(self.integral,-self.ilimit, self.ilimit)
        self.out += self.ki*self.integral

        # limit the total output
        if self.outlimit !=0:
            self.out = np.clip(self.out, -self.outlimit, self.outlimit)

        return self.out + self.bias
