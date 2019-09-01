'''Class to represent a PID policy

Has a bunch of different "branches" to distinguish the different kinds of policies we can make with PID outputs
Attributes: has a bunch of initialized PIDs
    - __init__ takes in a mode and creates PIDs based on the mode that is given
Methods:
    - chooseAction: outputs PWM inputs based on current state and PID list
    - update: updates the PIDs with given new states...has to be inputted in certain order

At the bottom, includes code using CrazyFlieSim.py to test given policy modes'''

from PID import PID
import torch
import numpy as np

class policy():
    def __init__(self, policyDict):
        self.mode = policyDict['PolicyMode']
        self.PID = []
        self.numPIDs = 0
        self.min_pwm = policyDict['min_pwm']
        self.max_pwm = policyDict['max_pwm']
        self.equil = policyDict['Equil']
        self.dt = policyDict['dt']
        self.numParameters = 0
        parameters = policyDict['PID']
        #order: pitch, roll, yaw, pitchrate, rollrate, yawRate or pitch roll yaw yawrate for hybrid or pitch roll yaw for euler
        if self.mode == 'EULER':
            self.numParameters = 9
        elif self.mode == 'HYBRID':
            self.numParameters = 12
        elif self.mode == 'RATE' or self.mode == 'ALL':
            self.numParameters = 18
        assert len(parameters) == self.numParameters
        self.numPIDs =int(self.numParameters / 3)

        for i in [3 * i for i in list(range(self.numPIDs))]:
            self.PID += [PID(0, parameters[i], parameters[i + 1], parameters[i + 2], 1000, self.dt)]


    def chooseAction(self):
        def limit_thrust(PWM): #Limits the thrust
            return np.clip(PWM, self.min_pwm, self.max_pwm)
        output = [0,0,0,0]
        #PWM structure: 0:front right  1:front left  2:back left   3:back right
        '''Depending on which PID mode we are in, output the respective PWM values based on PID updates'''
        if self.mode == 'EULER':
            output[0] = limit_thrust(self.equil[0] - self.PID[0].out + self.PID[1].out + self.PID[2].out)
            output[1] = limit_thrust(self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[2].out)
            output[2] = limit_thrust(self.equil[2] + self.PID[0].out - self.PID[1].out + self.PID[2].out)
            output[3] = limit_thrust(self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[2].out)
        elif self.mode == 'HYBRID':
            output[0][0] = limit_thrust(self.equil[0] - self.PID[0].out + self.PID[1].out + self.PID[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[5].out)
            output[0][2] = limit_thrust(self.equil[2] + self.PID[0].out - self.PID[1].out + self.PID[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[5].out)
        elif self.mode == 'RATE': #update this with the signs above
            output[0][0] = limit_thrust(self.equil[0] + self.PID[3].out - self.PID[4].out + self.PID[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.PID[3].out - self.PID[4].out - self.PID[5].out)
            output[0][2] = limit_thrust(self.equil[2] - self.PID[3].out + self.PID[4].out + self.PID[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.PID[3].out + self.PID[4].out - self.PID[5].out)
        elif self.mode == 'ALL': #update this with the signs above
            output[0][0] = limit_thrust(self.equil[0] + self.PID[0].out - self.PID[1].out + self.PID[2].out + self.PID[3].out - self.PID[4].out + self.PID[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[2].out - self.PID[3].out - self.PID[4].out - self.PID[5].out)
            output[0][2] = limit_thrust(self.equil[2] - self.PID[0].out + self.PID[1].out + self.PID[2].out - self.PID[3].out + self.PID[4].out + self.PID[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[2].out + self.PID[3].out + self.PID[4].out - self.PID[5].out)
        return torch.FloatTensor(output)

    def update(self, states):
        '''Order of states being passed: pitch, roll, yaw'''
        '''Updates the PID outputs based on the states being passed in (must be in the specified order above)'''
        '''Order of PIDs: pitch, roll, yaw, pitchRate, rollRate, yawRate'''
        assert len(states) == 3
        EulerOut = [0,0,0]
        for i in range(3):
            EulerOut[i] = self.PID[i].update(states[i])
        if self.mode == 'HYBRID':
            self.PID[3].update(EulerOut[2])
        if self.mode == 'RATE' or self.mode == 'ALL':
            for i in range(3):
                self.PID[i + 3].update(EulerOut[i])
