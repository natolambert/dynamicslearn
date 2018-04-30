from dynamics import *
from controllers import *
from dynamics_ionocraft import IonoCraft
from utils_plot import *
from utils_data import *
from models import NeuralNet

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable


dt = .01
iono1 = IonoCraft(dt, x_noise = .001)

m = 67e-6
g = 9.81

mgo4 = m*g/4

x0 = np.zeros(12)
u0 = np.array([mgo4,mgo4,mgo4,mgo4]) #np.zeros(4)

x1 = iono1.simulate(x0,u0)
x2 = iono1.simulate(x1,u0)
print('Original Test Simulation Steps')
print(x1)
print(x2)
print('\n')

# Generate lightly rnadom trajectory for plotting
# t = 0.5
# n = int(t/dt)
# T = np.linspace(0,t,n+1)
# X = np.array([x0])
# for i in range(n):
#     u = np.array([mgo4,mgo4,mgo4,mgo4]) + np.random.normal(scale = .0000005, size=4)
#     x0 = X[-1]
#     xnew = iono1.simulate(x0,u)
#     X = np.append(X, [xnew],axis=0)

length = 50
num_iter = 100
rand1 = randController(iono1, variance = .000005)
#[x y z xdot ydot zdot psi theta phi psi_dot theta_dot phi_dot]
X, U = generate_data(iono1, sequence_len=length, num_iter=num_iter, controller = rand1)
X = np.array(X)
U = np.array(U)
#translating from [psi theta phi] to [sin(psi)  sin(theta) sin(phi) cos(psi) cos(theta) cos(phi)]
# modX = np.concatenate((X[:, :, 0:3], np.sin(X[:, :, 3:6]), np.cos(X[:, :, 3:6]), X[:, :, 6:]), axis=2)

# #Getting output dX
# dX = np.array([states2delta(val) for val in modX])

# #the last state isn't actually interesting to us for training, as we only need dX
# #Follow by flattening the matrices so they look like input/output pairs
# modX = modX[:, :-1, :]
# modX = modX.reshape(modX.shape[0]*modX.shape[1], -1)

# modU = U[:, :-1, :]
# modU = modU.reshape(modU.shape[0]*modU.shape[1], -1)

# dX = dX.reshape(dX.shape[0]*dX.shape[1], -1)

# #at this point they should look like input output pairs
# if dX.shape != modX.shape:
# 	raise ValueError('Something went wrong, modified X shape:' + str(modX.shape) + ' dX shape:' + str(dX.shape))


# #Getting standard scalars for X, U, dX
# scalarX = normalize_states(modX)
# scalarU = normalize_states(modU)
# scalardX = normalize_states(dX)


# normX = scalarX.transform(modX)
# normU = scalarU.transform(modU)
# normdX = scalardX.transform(dX)

# inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
# outputs = torch.Tensor(dX)

# #creating neural network with 2 layers of 100 linearly connected ReLU units

nn = NeuralNet([19, 100, 100, 15])

# acc = nn.train(list(zip(inputs, outputs)), learning_rate=1e-4, epochs=100)

acc = nn.train((X, U), learning_rate=1e-4, epochs=50)

dx0 = nn.predict(x0, u0)
print("x0 + dx0=" + str(x0 + dx0), "\n x1=" + str(x1))
print("loss =" + str(sum((x1 - (x0 + dx0))**2)))
print("x0 + dx0 - x1=" + str(x0 + dx0 - x1))

print("\n \n \n ")

dx1 = nn.predict(x1, u0)
print("x1 + dx1=" + str(x1 + dx1), "\n x2=" + str(x2))
print("loss =" + str(sum((x2 - (x1 + dx1))**2)))
print("x1 + dx1 - x2=" + str(x1 + dx1 - x2))


print("\n \n \n")

plt.plot((x2 - (x1 + dx1))**2, 'bo')
print("element-wise loss:" + str((x2 - (x1 + dx1))**2))

plt.figure()
plt.plot(acc) #plotting accuracy of neural net as a function of training
plt.show()




