# Import project files
import utils_data

# Import External Packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

# torch packages
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Swish import Swish


from lossfnc_pnngaussian import PNNLoss_Gaussian

class GeneralNN(nn.Module):
    def __init__(self,
        n_in_input,
        n_in_state,
        n_out,
        state_idx_l,
        prob = True,
        hidden_w = 300,
        input_mode = 'Trajectories',
        pred_mode = 'Next State',
        ang_trans_idx = [],
        depth = 2,
        activation = "ReLU",
        B = 1.0):

        super(GeneralNN, self).__init__()
        """
        Simpler implementation of my other neural net class. After parameter tuning, now just keep the structure and change it if needed. Note that the data passed into this network is only that which is used.

        Parameters:
         - prob - If it is probablistic, True, else becomes deterministic three_input - if true, takes inputs of form [Thrust, tau_x, tau_y] rather than [F1, F2, F3, F4]
         - hidden_w - width of the hidden layers of the NN
         - n_in_state, n_in_input, n_out are all lengths of the inputs and outputs of the neural net
         - input_mode - either 'Trajectories' or else is a long list of recorded data, with separate snippets separated by a row of 0s
         - pred_mode - either 'Next State' or 'Delta State' and changes whether the NN is trained on x_{t+1} = f(x_t,u_t) (next state) or x_{t+1} = x_t + f(x_t,u_t) (delta state)
         - state_idx_l - list in order of passed states that is their positions in a full state vector
         - ang_trans_idx - list of indices of the inputed states that we want to transform as cosine(x) and sine(x) as we pass through the NN. eg. if the passed state is [roll, pitch, accelxyz], this list will be [0,1], and the input and output will be transformed to [sin(roll), cos(roll), sin(pitch), cos(pitch), accelxyz]. Outputs changed proportionally, but we keep track of this, because on the output side, we get [arctan2(sin(roll),cos(roll)), arctan2(sin(pitch),cos(pitch)), accelxyz]

        """
        # Store the parameters:
        self.prob = prob
        self.hidden_w = hidden_w
        self.n_in_input = n_in_input
        self.n_in_state = n_in_state
        self.n_in = n_in_input + n_in_state
        self.n_out = n_out
        self.input_mode = input_mode
        self.pred_mode = pred_mode
        self.ang_trans_idx = ang_trans_idx
        self.state_idx_l = state_idx_l
        self.depth = depth
        self.activation = activation
        self.B = B
        # increases number of inputs and outputs if cos/sin is used
        # plus 1 per angle because they need a pair (cos, sin) for each output
        if len(self.ang_trans_idx) > 0:
            self.n_in += len(self.ang_trans_idx)
            self.n_out += len(self.ang_trans_idx)

        #To keep track of what the mean and variance are at all times for transformations. Scalar is passed in init()
        #self.scalarX = MinMaxScaler()
        #self.scalarU = MinMaxScaler()
        #self.scalardX = MinMaxScaler()

        self.scalarX = RobustScaler()
        self.scalarU = RobustScaler()
        self.scalardX = RobustScaler()
        # Sets loss function
        if prob:
            self.loss_fnc = PNNLoss_Gaussian
        else:
            self.loss_fnc = nn.MSELoss

        # Probablistic nueral networks have an extra output for each prediction parameter to track variance
        if prob:
            self.n_out *= 2

        # Sequential object of network
        # The last layer has double the output's to include a variance on the estimate for every variable
        
        if self.activation == "ReLU":
            if self.depth == 1:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 2:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 3:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 4:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, hidden_w),
                    nn.ReLU(),
                    nn.Linear(hidden_w, self.n_out)
                )
        elif self.activation == "Swish":
            if self.depth == 1:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 2:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 3:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, self.n_out)
                )
            elif self.depth == 4:
                self.features = nn.Sequential(
                    nn.Linear(self.n_in, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, hidden_w),
                    Swish(self.B),
                    nn.Linear(hidden_w, self.n_out)
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
        # dX = np.array([utils_data.states2delta(val) for val in X])
        if self.input_mode == 'Trajectories':
            # if passed trajectories rather than a 2d array
            n, l, dx = np.shape(X)
            _, _, du = np.shape(U)

            # calcualte dX on trajectories rather than stacked elements
            if self.pred_mode == 'Next State':
                dX = X[:,1:,:]#-X[:,:-1,:]
            else:
                dX = X[:,1:,:] - X[:,:-1,:]

            # Ignore last element of X and U sequences because do not see next state
            X = X[:,:-1,:]
            U = U[:,:-1,:]

            # reshape
            X = X.reshape(-1, dx)
            U = U.reshape(-1, du)
            dX = dX.reshape(-1, dx)
            # # If there are angles to transform, do that now before normalization
            # if len(self.ang_trans_idx > 0):
            #     X_angled_part = np.concatenate((
            #         np.sin(X[:,self.ang_trans_idx]), np.cos(X[:,self.ang_trans_idx])),axis=1)
            #     X_no_trans = [X[:,i] for i in range(dx) if i not in self.ang_trans_idx]
            #     X = np.concatenate((X_angled_part,X_no_trans[0].T),axis=1)
            #
            #     # Cannot do this preemptively on the Xs in trajectory mode (at least not easily)
            #     dX_angled_part = np.concatenate((
            #         np.sin(dX[:,self.ang_trans_idx]), np.cos(dX[:,self.ang_trans_idx])),axis=1)
            #     dX_no_trans = [dX[:,i] for i in range(dx) if i not in self.ang_trans_idx]
            #     dX = np.concatenate((dX_angled_part,dX_no_trans[0].T),axis=1)
            #
            # # TODO: Change the trajectories form to just reformat to the two column mode my appending zeros to the end of each trajectory. It is a little bit easier to debug when they are separate
        else:
            # ELSE: already 2d form, assumed the next state is removed. Assumed that dX can be calculated nicely
            # dX = X[1:,:] - X[:-1,:]
            # At the end of each trajectory a line of zeros
            n, dx = np.shape(X)
            _, du = np.shape(U)

            if self.pred_mode == 'Next State':
                dX = X[1:,:]#-X[:,:-1,:]
            else:
                dX = X[1:,:] - X[:-1,:]

            # np.where returns true when there are nonzero elements
            dX = dX[np.where(X.any(axis=1))[0]]
            U = U[np.where(X.any(axis=1))[0]]
            X = X[np.where(X.any(axis=1))[0]]

            '''
            breakpoints = find rows of 0
            dX = dX[np.where(X.any(axis=1))[0]]
            U = U[np.where(X.any(axis=1))[0]]
            X = X[np.where(X.any(axis=1))[0]]

            if each trajectory is appended with a row of 0s, the breakpoints are at line n and n-1
            '''

        # If there are angles to transform, do that now before normalization
        if (len(self.ang_trans_idx) > 0):
            X_angled_part = np.concatenate((
                np.sin(X[:,self.ang_trans_idx]), np.cos(X[:,self.ang_trans_idx])),axis=1)
            X_no_trans = np.array([X[:,i] for i in range(dx) if i not in self.ang_trans_idx])
            if len(self.ang_trans_idx) == dx:
                X = X_angled_part
            else:
                X = np.concatenate((X_angled_part,X_no_trans.T),axis=1)

            # doing this preemptively made more issues. Post process dX here too
            dX_angled_part = np.concatenate((
                np.sin(dX[:,self.ang_trans_idx]), np.cos(dX[:,self.ang_trans_idx])),axis=1)
            dX_no_trans = [dX[:,i] for i in range(dx) if i not in self.ang_trans_idx]
            if len(self.ang_trans_idx) == dx:
                dX = dX_angled_part
            else:
                dX = np.concatenate((dX_angled_part,dX_no_trans[0].T),axis=1)

        # print(np.shape(X))
        # print(np.shape(dX))
        #at this point they should look like input output pairs
        if dX.shape != X.shape:
            raise ValueError('Something went wrong, modified X shape:' + str(dX.shape) + ' dX shape:' + str(X.shape))

        #update mean and variance of the dataset with each training pass
        #self.scalarX.partial_fit(X)
        #self.scalarU.partial_fit(U)
        #self.scalardX.partial_fit(dX)

        self.scalarX.fit(X)
        self.scalarU.fit(U)
        self.scalardX.fit(dX)

        import matplotlib.pyplot as plt
        #plt.hist(dX[:,0], bins=100)
        #plt.hist(dX[:,1], bins=100)
        #plt.hist(dX[:,2], bins=100)
        #plt.hist(dX[:,3], bins=100)
        #plt.hist(dX[:,4], bins=100)
        #plt.show()
        #plt.hist(U[:,0], bins=100)
        #plt.hist(U[:,1], bins=100)
        #plt.hist(U[:,2], bins=100)
        #plt.hist(U[:,3], bins=100)
        #plt.show()
        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(X)
        normU = self.scalarU.transform(U)
        normdX = self.scalardX.transform(dX)

        #print("normalized dX: ")
        #for i in normdX: print(i)
        #print("normalized U : ")
        #for i in normU: print(i)


        #plt.hist(normdX[:,0], bins=100)
        #plt.hist(normdX[:,1], bins=100)
        #plt.hist(normdX[:,2], bins=100)
        #plt.hist(normdX[:,3], bins=100)
        #plt.hist(normdX[:,4], bins=100)
        #plt.hist(normdX[:,5], bins=100)
        #plt.show()
        #plt.hist(normU[:,0], bins=100)
        #plt.hist(normU[:,1], bins=100)
        #plt.hist(normU[:,2], bins=100)
        #plt.hist(normU[:,3], bins=100)
        #plt.show()
        #quit()
        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(normdX)

        return list(zip(inputs, outputs))

    def getNormScalers(self):
        return self.scalarX, self.scalarU, self.scalardX


    def postprocess(self, dX):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
        """
        # de-normalize so to say
        dX = self.scalardX.inverse_transform(dX.reshape(1,-1))
        dX = dX.ravel()
        # If there are angles to transform, do that now after re normalization in post processing
        if (len(self.ang_trans_idx) > 0):
            # for i in self.ang_trans_idx:
            dX_angled = [np.arctan2(dX[idx+j], dX[idx+j+1]) for (j,idx) in enumerate(self.ang_trans_idx)]
            if len(dX)/2 > 2*len(self.ang_trans_idx):
                dX_not = dX[2*len(self.ang_trans_idx):]
                dX = np.concatenate((dX_angled,dX_not))
            else:
                dX = dX_angled

        return np.array(dX)


    def train(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=PNNLoss_Gaussian(), split=0.8, preprocess=True):
        """
        usage:
        data = (X[::samp,ypr], U[::samp,:])
        or
        data = ((Seqs_X[:,::samp,ypr], Seqs_U[:,::samp,:]))

        acc = newNN.train(data, learning_rate=2.5e-5, epochs=150, batch_size = 100, optim="Adam")


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
            # print('Shape of dataset is:', len(dataset))

        if self.prob:
            loss_fn = PNNLoss_Gaussian()
        else:
            loss_fn = MSELoss()

        # makes sure loss fnc is correct
        if loss_fn == PNNLoss_Gaussian() and not self.prob:
            raise ValueError('Check NN settings. Training a deterministic net with pnnLoss. Pass MSELoss() to train()')

        trainLoader = DataLoader(dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)
        
        self.testData = dataset[int(split*len(dataset)):]

        #Unclear if we should be using SGD or ADAM? Papers seem to say ADAM works better
        if(optim=="Adam"):
            optimizer = torch.optim.Adam(super(GeneralNN, self).parameters(), lr=learning_rate)
        elif(optim=="SGD"):
            optimizer = torch.optim.SGD(super(GeneralNN, self).parameters(), lr=learning_rate)
        else:
            raise ValueError(optim + " is not a valid optimizer type")
        return self._optimize(loss_fn, optimizer, epochs, batch_size, trainLoader, testLoader)


    def predict(self, X, U):
        """
        Given a state X and input U, predict the change in state dX. This function is used when simulating, so it does all pre and post processing for the neural net
        """
        dx = len(X)
        # angle transforms
        if (len(self.ang_trans_idx) > 0):
            X_angled_part = np.concatenate((
                np.sin(X[self.ang_trans_idx]), np.cos(X[self.ang_trans_idx])))
            X_no_trans = np.array([X[i] for i in range(dx) if i not in self.ang_trans_idx])
            if len(self.ang_trans_idx) == dx:
                X = X_angled_part
            else:
                X = np.concatenate((X_angled_part,X_no_trans.T))

        #normalizing and converting to single sample
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))

        input = Variable(torch.Tensor(np.concatenate((normX, normU), axis=1)))

        NNout = self.forward(input).data[0]

        # If probablistic only takes the first half of the outputs for predictions
        if self.prob:
            NNout = self.postprocess(NNout[:int(self.n_out/2)]).ravel()
        else:
            NNout = self.postprocess(NNout).ravel()
        
        return NNout

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
                avg_loss += loss.item()/num_batches                  # update the overall average loss with this batch's loss

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
                test_error += loss.item()
            test_error = test_error / len(testLoader)



            #print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]),
            #          "test_error={:.9f}".format(test_error))
            print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(avg_loss.data[0]), "test loss=", "{:.6f}".format(test_error))
            errors.append(test_error)
        #loss_fn.print_mmlogvars()
        return errors

    def save_model(self, filepath):
        torch.save(self, filepath)                  # full model state
        # For load, use torch.load()

def predict_nn(model, x, u, indexlist):
    '''
    special, generalized predict function for the general nn class in construction.
    x, u are vectors of current state and input to get next state or change in state
    indexlist is is an ordered index list for which state variable the indices of the input to the NN correspond to. Assumes states come before any u
    '''
    # constructs input to nn
    #x_nn = []
    #for idx in indexlist:
    #    x_nn.append(x[idx])
    #x_nn = np.array(x_nn)
   
    # Makes prediction for either prediction mode. Handles the need to only pass certain states
    prediction = np.copy(x)
    pred_mode = model.pred_mode
    if pred_mode == 'Next State':
        pred = model.predict(x,u)
        for i, idx in enumerate(indexlist):
            prediction[idx] = pred[i]
    else:
        pred = model.predict(x,u)
        for i, idx in enumerate(indexlist):
            #print('x_nn = ', x[idx], 'predicted', pred)
            prediction[idx] = x[idx] + pred[i]
            #print('prediction list 1: ', prediction[0])
    #print('prediction list 2: ', prediction[0])


    return prediction
