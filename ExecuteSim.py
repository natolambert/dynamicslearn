'''Wrapper function that takes inputs of PID parameters, creates a policy, and runs a simulation with visuals on the policy.
   All simulations are based on outputs from CrazyFlieSim.py and noise can be added if need be. '''

from CrazyFlieSim import CrazyFlie
from PIDPolicy import policy
from ExecuteTrain import getState
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

'''Setting PID parameters to test, frequency, initial conditions, and dictionaries for parameters'''
#insert PID parameters here...order: pitch roll yaw prate rrate yrate each in PID order
#PID = [299.999, 3.182e-6, 2.3965511, 299.99999, 99.9999, 16.40728, 299.9999, 7.3238e-6, 19.67648]
PID = [19.55158993932205, 5.925000272611056, 2.7095558520976946, 16.556410900302456, 8.853667539101147, 0.0004685617828592727, 20.41008459214832, 3.676864181770996, 4.155875802638022]
dt = 1/75 #75 Hertz frequency
EQUILIBRIUM = [30000, 30000, 30000, 30000]
states = getState()
simulator = CrazyFlie(dt)
policyDict = { 'PID': PID,
               'PolicyMode':'EULER',
               'dt': dt,
               'Equil': EQUILIBRIUM,
               'min_pwm': 20000,
               'max_pwm': 65500}
PIDpolicy = policy(policyDict)

simulatorDict = {'policy': PIDpolicy,
                 'initStates': states,
                 'simFlights': 1,
                 'visuals': False,
                 'maxRollout': 10000,
                 'addNoise': True,
                 'maxAngle': 30}

'''Testing the policy in the simulator and printing results'''

totTime, avgTime, varTime, avgError = simulator.test_policy(simulatorDict)

print("Results:")
print("Total Time: ", totTime, " with maximum total time of ", simulatorDict['simFlights'] * simulatorDict['maxRollout'])
print("Average Time: ", avgTime, " with maximum average time of ", simulatorDict['maxRollout'])
print("Standard Deviation of time: ", varTime**(1/2), " over ", simulatorDict['simFlights'] ," iterations.")
print("Average error: ", avgError, " with maximum error of ", 3 * math.radians(simulatorDict['maxAngle']) * simulatorDict['maxRollout'])
