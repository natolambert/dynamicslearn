# File to run experiments with a learned model

from dynamics import *
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from utils_data import *
from models import *

import torch
import time

# Packages for Serial
import serial
import sys
from serial.tools.list_ports import comports as list_comports

cur_time = time.time()

# Define some parameters
N_mpc = 75              # Number of MPC actions to look through
T_mpc = 3               # Time horizon of mpc
dt_x = .0002        # 5khz dynam update
dt_m = .002         # 500 hz measurement update
dt_u = .004         # ~200 hz control update

# Init serial
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyUSB1',
    baudrate=9600,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_TWO,
    bytesize=serial.SEVENBITS
)

if not ser.isOpen():
    ser.open()

# Create dummy dynamics object to pass some values
iono1 = IonoCraft(dt_x)

# change equilibrium input if we need to

# Load model
nn = torch.load('expnn.pth')

# Objective functions
origin_minimizer = Objective(np.linalg.norm, 'min', 6, dim_to_eval=[6,7,8,12,13,14])

# Initialize mpc, dt_u is passed twice so that the mpc returns a control on every update
mpc1 = MPController(nn, iono1, dt_u, dt_u, origin_minimizer, N=N_mpc, T=T_mpc, variance = .00003)
print('...MPC Running')

# Run mpc on robot to  make it hover
try:
    while True:

        # Read current state from serial
        cur_state = ser.read()
        # convert string -> values

        # update MPC
        cur_control = mpc1.update(cur_state)

        # send control command
        ser.write(cur_control)

        print('Update time is: ', time.time()-cur_time)
        cur_time = time.time()
except:
    ser.close()
    print('Testing Ended')
