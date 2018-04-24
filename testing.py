from dynamics import *
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from models import LeastSquares

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dt = .01
iono1 = IonoCraft(dt, x_noise = .001)

m = 67e-6
g = 9.81

mgo4 = m*g/4

x0 = np.zeros(12)
u0 = np.array([mgo4,mgo4,mgo4,mgo4]) #np.zeros(4)

x1 = iono1.simulate(x0,u0)
x2 = iono1.simulate(x1,u0)

hover = HoverPID(iono1)
out = hover.update(x0)
# print(out)
# print(mgo4)
print('TESTING CRAZYFLIE')
quad = CrazyFlie(dt, x_noise = 0)
mgo4quad = quad.m*quad.g
u0 = np.array([mgo4quad,0,-.1,.1]) #np.zeros(4)
x1 = quad.update(x0,u0)
print(x1)
