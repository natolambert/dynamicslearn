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
        self.model = []
        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i+1]))
            self.model.append(nn.ReLU())
        del(self.model[-1])


    """
    Standard forward function necessary if extending nn.Module. Basically a copy of nn.Sequential
    """
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    """
    Train the neural network. 
    dataset is a list of tuples to train on, where the first value in the tuple is the training data (should be implemented as a torch tensor), and the second value in the tuple
    is the label/action taken
    Epochs is number of times to train on given training data, 
    batch_size is hyperparameter dicating how large of a batch to use for training, 
    optim is the optimizer to use (options are "Adam", "SGD")
    """
    def train(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=nn.MSELoss()):

        dataLoader = DataLoader(dataset, batch_size=batch_size)
        #Unclear if we should be using SGD or ADAM? Papers seem to say ADAM works better
        if(optim=="Adam"):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif(optim=="SGD"):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(optim + " is not a valid optimizer type")
        return _optimize(self.model, loss_fn, optimizer, epochs, batch_size, dataLoader)
    
    """
    Don't actually need this, but keeping it to be consistent with Least Squares class
    """
    def predict(self, D_in):
        return self.forward(D_in)



    def _optimize(self, loss_fn, optim, epochs, batch_size, trainloader):
        for epoch in range(epochs):
            avg_loss = Variable(torch.zeros(1))
            num_batches = len(trainset)/batch_size
            for i, (input, target) in enumerate(trainloader):
                input = Variable(input.view(batch_size, -1))  # the input comes as a batch of 2d images which we flatten;
                                                              # view(-1) tells pytorch to fill in a dimension; here it's 784
                optim.zero_grad()                             # zero the gradient buffers
                output = self.forward(input)                 # compute the output
                loss = loss_fn(target, output)                # compute the loss
                loss.backward()                               # backpropagate from the loss to fill the gradient buffers
                optim.step()                                  # do a gradient descent step
                if not loss.data.numpy() == loss.data.numpy(): # Some errors make the loss NaN. this is a problem.
                    print("loss is NaN")                       # This is helpful: it'll catch that when it happens,
                    return output, input, loss                 # and give the output and input that made the loss NaN
                avg_loss += loss/num_batches                  # update the overall average loss with this batch's loss
            
            
            """
            correct = 0
            for input, target in testset:                     # compute the testing accuracy
                input = Variable(input.view(1, -1))
                output = prediction_fn(input)
                pred_ind = torch.max(output, 1)[1]              
                if pred_ind.data[0] == target:                # true/false Variables don't actually have a boolean value,
                    correct += 1                              # so we have to unwrap it to see if it was correct
            accuracy = correct/len(testset)
            """
            #print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]),
            #          "accuracy={:.9f}".format(accuracy))
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]))
        return acc

#
# class GaussianProcess:
#     # TODO

def simulate_learned(model, actions, x0=[]):
    # returns a vector of the states predicted by the learned dynamics model given states and the inputs
    raise NotImplementedError('To Be Done, SOON')
