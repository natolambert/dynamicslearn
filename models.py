# Compatibility Python 3

# Import project files
import utils_data

# Import External Packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# torch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LeastSquares:
    # fits gathered data to the form
    # x_(t+1) = Ax + Bu

    def __init__(self, x_dim = 12, u_dim = 4):
        self.reg = linear_model.LinearRegression()
        self.x_dim = x_dim
        self.u_dim = u_dim

    def train(self, change_states, states_prev, actions_prev):
        # need to make sure data here is normalized AND the states are all
        # formed as change of state, rather than arbitary values (this makes sure
        # that the features are not fit to large vlaues)

        # Xw = y
        # this works if we stack a array
        #       z = [x, u]
        # then can make a matrix
        #       w ~ [A, B]
        Z = np.hstack([states_prev, actions_prev])
        y = change_states

        self.reg.fit(Z,y)
        return self.reg.coef_

    def predict(self, state, action):
        # predicts next state of a state, action pairing

        # forces one dimensional vector, transpose to allign with .fit dimensions
        vect = np.hstack((state, action)).reshape(-1,1).T
        pred = self.reg.predict(vect)
        return pred[0]

    @property
    def A_B(self):
        # function that prints a readable form
        print('Not Implemented lol')

class NeuralNet(nn.Module):
    # NOTE

    """
    Assumes standard Linear input + output and ReLU activations for all hidden layers
    Layers is a list of layer sizes, first layer size should be input dimension, last layer size should be output dimension
    """
    def __init__(self, layers):
        super(NeuralNet, self).__init__()

        #To keep track of what the mean and variance are at all times for transformations
        self.scalarX = StandardScaler()
        self.scalarU = StandardScaler()
        self.scalardX = StandardScaler()

        for i in range(len(layers) - 1):
            self.add_module(str(2*i), nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                self.add_module(str(2*i+1), nn.ReLU())


    """
    Standard forward function necessary if extending nn.Module. Basically a copy of nn.Sequential
    """
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    """
    Preprocess X and U (as they would be outputted by dynamics.generate_data) so they can be passed into the neural network for training
    X and U should be numpy arrays with dimensionality X.shape = (num_iter, sequence_len, 12) U.shape = (num_iter, sequence_len, 4)

    return: A
    """
    def preprocess(self, X, U):
        #translating from [psi theta phi] to [sin(psi)  sin(theta) sin(phi) cos(psi) cos(theta) cos(phi)]
        modX = np.concatenate((X[:, :, 0:3], np.sin(X[:, :, 3:6]), np.cos(X[:, :, 3:6]), X[:, :, 6:]), axis=2)

        #Getting output dX
        dX = np.array([utils_data.states2delta(val) for val in modX])

        #the last state isn't actually interesting to us for training, as we only train (X, U) --> dX
        modX = modX[:, :-1, :]
        modU = U[:, :-1, :]
        
        #Follow by flattening the matrices so they look like input/output pairs
        modX = modX.reshape(modX.shape[0]*modX.shape[1], -1)
        modU = modU.reshape(modU.shape[0]*modU.shape[1], -1)
        dX = dX.reshape(dX.shape[0]*dX.shape[1], -1)

        #at this point they should look like input output pairs
        if dX.shape != modX.shape:
            raise ValueError('Something went wrong, modified X shape:' + str(modX.shape) + ' dX shape:' + str(dX.shape))
        
        #update mean and variance of the dataset with each training pass
        self.scalarX.partial_fit(modX)
        self.scalarU.partial_fit(modU)
        self.scalardX.partial_fit(dX)

        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(modX)
        normU = self.scalarU.transform(modU)
        normdX = self.scalardX.transform(dX)

        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(dX)

        return list(zip(inputs, outputs))



    """
    Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset and converting from cos, sin
    to actual angle
    """
    def postprocess(self, dX):
        dX = self.scalardX.inverse_transform(dX)
        if(len(dX.shape) > 1):
            out = np.concatenate((dX[:, :3], np.arctan2(dX[:, 3:6], dX[:, 6:9]), dX[:, 9:12], dX[:, 12:]), axis=1)
        else:
            out = np.concatenate((dX[:3], np.arctan2(dX[3:6], dX[6:9]), dX[9:12], dX[12:]))
        return out
    """
    Train the neural network. 
    if preprocess = False
        dataset is a list of tuples to train on, where the first value in the tuple is the training data (should be implemented as a torch tensor), and the second value in the tuple
        is the label/action taken
    if preprocess = True
        dataset is simply the raw output of generate data (X, U)
    Epochs is number of times to train on given training data, 
    batch_size is hyperparameter dicating how large of a batch to use for training, 
    optim is the optimizer to use (options are "Adam", "SGD")
    split is train/test split ratio
    """
    def train(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=nn.MSELoss(), split=0.9, preprocess=True):
        if preprocess:
            dataset = self.preprocess(dataset[0], dataset[1])

        trainLoader = DataLoader(dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)

        #Unclear if we should be using SGD or ADAM? Papers seem to say ADAM works better
        if(optim=="Adam"):
            optimizer = torch.optim.Adam(super(NeuralNet, self).parameters(), lr=learning_rate)
        elif(optim=="SGD"):
            optimizer = torch.optim.SGD(super(NeuralNet, self).parameters(), lr=learning_rate)
        else:
            raise ValueError(optim + " is not a valid optimizer type")
        return self._optimize(loss_fn, optimizer, epochs, batch_size, trainLoader, testLoader)
    
    """
    Given a state X and input U, predict the change in state dX. This function does all pre and post processing for the neural net
    """
    def predict(self, X, U):
        #Converting to sin/cos form
        X = np.concatenate((X[0:3], np.sin(X[3:6]), np.cos(X[3:6]), X[6:]))
        
        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))

        input = Variable(torch.Tensor(np.concatenate((normX, normU), axis=1)))

        merp = self.forward(input)
        print(merp)

        return self.postprocess(self.forward(input).data[0])



    def _optimize(self, loss_fn, optim, epochs, batch_size, trainLoader, testLoader):
        errors = []
        for epoch in range(epochs):
            avg_loss = Variable(torch.zeros(1))
            num_batches = len(trainLoader)/batch_size
            for i, (input, target) in enumerate(trainLoader):
                #input = Variable(input.view(batch_size, -1))  # the input comes as a batch of 2d images which we flatten;
                                                              # view(-1) tells pytorch to fill in a dimension; here it's 784
                input = Variable(input)
                target = Variable(target, requires_grad=False) #Apparently the target can't have a gradient? kinda weird, but whatever
                optim.zero_grad()                             # zero the gradient buffers
                output = self.forward(input)                 # compute the output
                loss = loss_fn(output, target)                # compute the loss
                loss.backward()                               # backpropagate from the loss to fill the gradient buffers
                optim.step()                                  # do a gradient descent step
                if not loss.data.numpy() == loss.data.numpy(): # Some errors make the loss NaN. this is a problem.
                    print("loss is NaN")                       # This is helpful: it'll catch that when it happens,
                    return output, input, loss                 # and give the output and input that made the loss NaN
                avg_loss += loss/num_batches                  # update the overall average loss with this batch's loss
            
            
            test_error = 0
            for (input, target) in testLoader:                     # compute the testing test_error
                input = Variable(input)
                target = Variable(target, requires_grad=False)
                output = self.forward(input)
                loss = loss_fn(output, target)
                test_error += loss.data[0]
            test_error = test_error / len(testLoader)
            
            #print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]),
            #          "test_error={:.9f}".format(test_error))
            print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(avg_loss.data[0]), "test loss=", "{:.6f}".format(test_error))
            errors.append(test_error)
        return errors

#
# class GaussianProcess:
#     # TODO

def simulate_learned(model, actions, x0=[]):
    # returns a array of the states predicted by the learned dynamics model given states and the inputs
    if (x0 == []):
        x0 = np.zeros(model.x_dim,1)

    X = [x0]
    for a in actions:
        # print(a)
        xnext = X[-1].flatten() + model.predict(X[-1], a)
        X.append(xnext)

    return np.array(X)
