import torch
from torch.nn import MSELoss

from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics import *
import numpy as np

from utils_data import *

########################################################################
testTime = 5000  	# Time to run realtime test, in ms


############################### Get NN Model ###########################
print('Loading offline model')
newNN = torch.load('_models/2018-06-26_general_temp.pth')


################################ Obj Fnc ################################
origin_minimizer = Objective(np.linalg.norm, 'min', 6, dim_to_eval=[6,7,8,12,13,14])
print('...Objective Function Initialized')

################################ MPC ####################################
mpc1 = MPController(newNN, iono1, dt_x, dt_u, origin_minimizer, N=50, T=5, variance = .00003)
print('...MPC Running')


######################## Controller Input/Output ##################
start_time = datetime.now()


while  millis() < testTime:
	# get current state
	x_prev = parseFlie   # WRITE DIS
	receieved = millis()

	# generate new u
	u = mpc1.update(x_prev)   # UPDATE SHOULD RETURN PWMs
	generated = millis()

	# pass new u to ROS node

	# print for debugging / viewing timestamps to measure control latency and update frequ.
	print (receieved, xprev, generated, u)


