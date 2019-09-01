import pickle
import numpy as np
from GenNN import GeneralNN
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import pickle
from sklearn.model_selection import KFold
from kMeansData import kClusters
from PNNLoss import PNNLoss_Gaussian
import matplotlib.pyplot as plt

class EnsembleNN(nn.Module):
    '''Ensemble neural network class
        Attributes:
            E: number of networks/ensembles
            prob: flag signaling PNN
            networks: list of networks/ensembles
            n_in_input: number of elements in PWM input vector
            n_in_state: number of elements in state input vector
            n_out: number of elements in output vector
            stack: size of stacking for inputs
            epsilon: small value to add to prevent Nan values
        Methods:
            train_cust: takes data and train parameters to train each network
            init_weights_orth: initializes all weights/matrices in nn to be orthonormal
            predict: makes averaged prediction of next state given a current state/action pair
            plot_ensembles: makes a plot of training loss and testing loss with respect to epochs
            save_model: saves model to a certain path location'''

    def __init__(self, nn_params, E = 5):
        super(EnsembleNN, self).__init__()
        self.E = E
        self.prob = nn_params[0]['bayesian_flag']
        self.networks = []
        self.n_in_input = nn_params[0]['du']
        self.n_in_state = nn_params[0]['dx']
        self.n_in = self.n_in_input + self.n_in_state
        self.n_out = nn_params[0]['dt']
        if self.prob:
            self.n_out = self.n_out * 2
        self.stack = nn_params[0]['stack']
        for i in range(self.E):
            self.networks.append(GeneralNN(nn_params[i]))
        self.epsilon = nn_params[0]['epsilon']


    def train_cust(self, dataset, train_params, numClusters = 5, cluster = False, gradoff = False, datasize = 0, padding = 0):
        #If we want to cluster the dataset into self.clusters clusters and from those clusters
        if cluster:
            print("")
            print("Clustering...")
            km = kClusters(min(np.shape(dataset[0])[0], numClusters), padding) #if number of datapoints > clusters, we have an issue. So put this in
            km.cluster(dataset) #performs clustering for us
            dataset, leftover = km.sample() #performs sampling returning new dataset and leftover dataset for testing
            if datasize != 0: #if we want a specific size of data for training set
                sizeTrain = min(datasize, dataset.shape[0])
                sizeTest = int(sizeTrain * (1 - train_params[0]["split"]) / (train_params[0]["split"])) #take the rest and put in test
                trainidx = np.random.choice(dataset.shape[0], sizeTrain)
                testidx = np.random.choice(leftover.shape[0], sizeTest)
                dataset = dataset[trainidx, :]
                leftover = leftover[testidx, :]
            lenTrain = dataset.shape[0]
            lenTest = leftover.shape[0]
            for dict in train_params: #fix the split of the training to reflect our sampled data set and leftovers
                dict["split"] =lenTrain / (lenTrain + lenTest)
            print("Used a split ratio of: ", lenTrain / (lenTrain + lenTest))
            print("")
            dataset = np.vstack((dataset, leftover))
            dataset = ((dataset[:, :self.n_in_state], dataset[:, self.n_in_state: self.n_in], dataset[:, self.n_in:])) #tune this to match data inputs
        TrTePairs = [] #training and testing loss pairs for each neural network in the ensemble
        mintrain = []
        mintest = []
        lastEpoch = None

        '''Training each of the neural networks on the same dataset with different parameters'''

        for (i, net) in enumerate(self.networks):
            #print("Training network number: ", i + 1)
            acctest, acctrain = net.train_cust(dataset, train_params[i], gradoff = False)
            if acctrain[-1] == float('nan'): #get rid of the last number if it is Nan
                TrTePairs += [[acctrain[:-1], acctest[:-1]]]
                mintrain += [min(acctrain[:-1])]
                mintest += [min(acctest[:-1])]
            else: #update the minimum training and testing loss lists
                TrTePairs += [[acctrain, acctest]]
                mintrain += [min(acctrain)]
                mintest += [min(acctest)]

        '''Displaying the results'''

        #self.plot_ensembles(TrTePairs)
        #print("")
        #print("")
        #print("RESULTS:")
        #for i in range(len(self.networks)):
            #print("Network number", i + 1, ":", " Minimum Testing Loss: ", mintest[i], " Minimum Training Loss: ", mintrain[i], " Epochs trained: ", len(TrTePairs[i][0]))
        #print("")
        mintest = sum(mintest) / (len(mintest))
        mintrain = sum(mintrain) / (len(mintrain))
        print("Overall: Average testing loss: ", mintest, " Average training loss: ", mintrain)

        return mintest, mintrain


    def init_weights_orth(self):
        for nn in self.networks:
            nn.init_weights_orth()

    def predict(self, X, U):
        #given an arbitrary number of input states and PWMs, output the predictions this model will give
        prediction = np.zeros((len(X), self.networks[0].n_out))

        for net in self.networks:
            for i in range(len(X)):
                mean, var = (net.predict(X[i, :], U[i, :]))
                prediction [i, :] += ((1/self.E) * np.hstack((mean,var)))

        return prediction #+ self.epsilon

    def plot_ensembles(self, pairs):
        #plots the resulting training and testing loss of each neural network in the ensemble
        for pair in pairs:
            #training is 0, testing is 1
            eps = list(range(1, len(pair[0]) + 1))
            plt.plot(eps, pair[0], 'r--', eps, pair[1], 'b--')
            plt.show()

    def save_model(self, filepath):
        #saves the model to a specified filepath
        torch.save(self, filepath)
        print("")
        print("EnsembleModel has been saved to " + filepath)
