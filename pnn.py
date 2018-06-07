# Import project files
import utils_data

# Import External Packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# torch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pnnloss_gaussian import PNNLoss_Gaussian

class PNeuralNet(nn.Module):
    def __init__(self):
        super(PNeuralNet, self).__init__()
        """
        Simpler implementation of my other neural net class. After parameter tuning, now just keep the structure and change it if needed.
        """

        #To keep track of what the mean and variance are at all times for transformations
        self.scalarX = MinMaxScaler()
        self.scalarU = MinMaxScaler()
        self.scalardX = MinMaxScaler()

        # Sequential object of network
        # The last layer has double the output's to include a variance on the estimate for every variable
        self.features = nn.Sequential(
            nn.Linear(12, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 18)
        )



    def forward(self, x):
        """
        Standard forward function necessary if extending nn.Module.
        """
        x = self.features(x)
        return x.view(x.size(0), -1)

    def preprocess(self, X, U):
        """
        Preprocess X and U for passing into the neural network. For simplicity, takes in X and U as they are output from generate data, but only passed the dimensions we want to prepare for real testing. This removes a lot of potential questions that were bugging me in the general implementation. Will do the cosine and sin conversions externally.
        """
        # Already done is the transformation from
        # [yaw, pitch, roll, x_ddot, y_ddot, z_ddot]  to
        # [sin(yaw), sin(pitch), sin(roll), cos(pitch), cos(yaw),  cos(roll), x_ddot, y_ddot, z_ddot]
        print(np.shape(X))
        dX = np.array([utils_data.states2delta(val) for val in X])

        # Take cosine transformation. Let's try this.
        X = np.concatenate((np.sin(X[:, :, :3]), np.cos(X[:, :, :3]), X[:, :, 3:]), axis=2)
        print(np.shape(X))
        dX = np.concatenate((np.sin(dX[:, :, :3]), np.cos(dX[:, :, :3]), dX[:, :, 3:]), axis=2)

        n, l, dx = np.shape(X)
        _, _, du = np.shape(U)
        #Getting output dX

        # Ignore last element of X and U sequences because do not see next state
        X = X[:,:-1,:]
        U = U[:,:-1,:]

        # Reshape all to shape (n-1)*l x d
        dX = dX.reshape(-1, dx)
        X = X.reshape(-1, dx)
        U = U.reshape(-1, du)

        #at this point they should look like input output pairs
        if dX.shape != X.shape:
            raise ValueError('Something went wrong, modified X shape:' + str(dX.shape) + ' dX shape:' + str(X.shape))

        print(np.shape(X))
        #update mean and variance of the dataset with each training pass
        self.scalarX.partial_fit(X)
        self.scalarU.partial_fit(U)
        self.scalardX.partial_fit(dX)

        print(np.shape(X))
        print(np.shape(U))

        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(X)
        normU = self.scalarU.transform(U)
        normdX = self.scalardX.transform(dX)

        # print(np.mean(normX))
        # print(np.mean(normU))
        # print(np.mean(normdX))

        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(normdX)

        return list(zip(inputs, outputs))

    def postprocess(self, dX):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
        """
        # de-normalize so to say
        dX = self.scalardX.inverse_transform(dX.reshape(1,-1))
        return np.array(dX)


    def train(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=PNNLoss_Gaussian(), split=0.8, preprocess=True):
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
        if preprocess:
            dataset = self.preprocess(dataset[0], dataset[1])
            print('Shape of dataset is:', len(dataset))

        trainLoader = DataLoader(dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)

        #Unclear if we should be using SGD or ADAM? Papers seem to say ADAM works better
        if(optim=="Adam"):
            optimizer = torch.optim.Adam(super(PNeuralNet, self).parameters(), lr=learning_rate)
        elif(optim=="SGD"):
            optimizer = torch.optim.SGD(super(PNeuralNet, self).parameters(), lr=learning_rate)
        else:
            raise ValueError(optim + " is not a valid optimizer type")
        return self._optimize(loss_fn, optimizer, epochs, batch_size, trainLoader, testLoader)


    def predict(self, X, U):
        """
        Given a state X and input U, predict the change in state dX. This function is used when simulating, so it does all pre and post processing for the neural net
        """

        # [yaw, pitch, roll, x_ddot, y_ddot, z_ddot]  to
        # [sin(yaw), cos(yaw), sin(pitch), cos(pitch), sin(roll), cos(roll), x_ddot, y_ddot, z_ddot]
        X = X[[6,7,8,12,13,14]]
        X = np.concatenate((np.sin(X[:3]), np.cos(X[:3]), X[3:]))

        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))

        input = Variable(torch.Tensor(np.concatenate((normX, normU), axis=1)))

        NNout = self.forward(input).data[0]

        # Takes first nine outputs that are the means of the probablistic neural network
        Means = self.postprocess(NNout[:9]).ravel()

        # Transform to sine and cosine. Out here is a length 6 vector
        # [yaw, pitch, roll, x_ddot, y_ddot, z_ddot]
        trans = np.concatenate((np.arctan2(Means[:3], Means[3:6]), Means[6:])).ravel()
        out = np.zeros(15)
        out[6:9] = trans[:3]
        out[12:] = trans[3:]
        return out

    def _optimize(self, loss_fn, optim, epochs, batch_size, trainLoader, testLoader):
        errors = []
        for epoch in range(epochs):
            avg_loss = Variable(torch.zeros(1))
            num_batches = len(trainLoader)/batch_size
            for i, (input, target) in enumerate(trainLoader):
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
                avg_loss += loss.data[0]/num_batches                  # update the overall average loss with this batch's loss

            # Debugging:
            # print('NN Output: ', output)
            # print('Target: ', target)
            # print(np.shape(output))
            # print(np.shape(target))

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

    def save_model(self, filepath):
        # torch.save(self.state_dict(), filepath)   # only param
        torch.save(self, filepath)                  # full model state
        # print(self.scalarX.get_params())

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
