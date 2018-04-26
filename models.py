# Compatibility Python 3

# Import project files

# Import External Packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# torch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LeastSquares:
    # fits gathered data to the form
    # x_(t+1) = Ax + Bu

    def __init__(self):
        self.reg = linear_model.LinearRegression()

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

        vect = np.hstack(state, action)
        pred = self.reg.predict(vect)

        return pred

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
    Train the neural network. 
    dataset is a list of tuples to train on, where the first value in the tuple is the training data (should be implemented as a torch tensor), and the second value in the tuple
    is the label/action taken
    Epochs is number of times to train on given training data, 
    batch_size is hyperparameter dicating how large of a batch to use for training, 
    optim is the optimizer to use (options are "Adam", "SGD")
    split is train/test split ratio
    """
    def train(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=nn.MSELoss(), split=0.9):

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
    Don't actually need this, but keeping it to be consistent with Least Squares class
    """
    def predict(self, D_in):
        return self.forward(D_in)



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
    # returns a vector of the states predicted by the learned dynamics model given states and the inputs
    raise NotImplementedError('To Be Done, SOON')
