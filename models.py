# Compatibility Python 3

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
from torch.utils.data.sampler import WeightedRandomSampler
import copy         # for copying models in ensembleNN

class LeastSquares:
    '''
    fits gathered data to the form
    x_(t+1) = Ax + Bu
    .train() to train the fit
    .predict() to predict the next state from the current state and inputs
    '''

    def __init__(self, dt_x, x_dim = 12, u_dim = 4):
        self.reg = linear_model.LinearRegression()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt_x = dt_x

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
        return self.dt_x*pred[0]

    @property
    def A_B(self):
        # function that prints a readable form
        print('Not Implemented lol')

class NeuralNet(nn.Module):
    """
    - layer_sizes is a list of layer sizes, first layer size should be input dimension, last layer size should be output dimension
    - layer_types is a list of activation functions for the middle layers. Note that current implementation sets the first layer to be linear regardless.
    - learn_list is a list of state variables to use in training the dynamics. The model will learn and predict this variables.
    """
    def __init__(self, layer_sizes, layer_types, dynam, state_learn_list, input_learn_list):
        super(NeuralNet, self).__init__()

        #To keep track of what the mean and variance are at all times for transformations
        # self.scalarX = StandardScaler()
        # self.scalarU = StandardScaler()
        # self.scalardX = StandardScaler()
        self.scalarX = MinMaxScaler(feature_range=(-1, 1))
        self.scalarU = MinMaxScaler(feature_range=(-1, 1))
        self.scalardX = MinMaxScaler(feature_range=(-1, 1))

        # list of states and inputs to learn dynamics from
        self.state_learn_list = state_learn_list
        self.input_learn_list = input_learn_list

        # dynam file for reference
        self.dynam = dynam

        if (len(layer_sizes) != len(layer_types)):
            raise ValueError('Number of layer sizes does not match number of layer types passed.')

        # num_angles = sum(1 for x in dynam.x_dict.values() if x[1] == 'angle')
        num_angles = 0
        for state in state_learn_list:
            key = dynam.x_dict[state]
            if key[1] == 'angle':
                num_angles +=1
        if ((len(state_learn_list)+len(input_learn_list)+num_angles) != layer_sizes[0]):
            raise ValueError('Dimension of states and inputs to learn from does not match the first layer dimension.')

        # Add linear layer with activations
        l=0             # label iterator
        for i in range(len(layer_sizes) - 1):

            # add linear layers of size [n_in, n_out]
            self.add_module(str(l), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            l+= 1

            # for all but last layer add activation function
            if (layer_types[i] != 'nn.Linear()'):
                if (layer_types[i] == 'nn.ReLU()'):
                    self.add_module(str(l), nn.LeakyReLU())
                    l =+ 1
                else:
                    raise ValueError('Layer Type Not Implemented')


    """
    Standard forward function necessary if extending nn.Module. Basically a copy of nn.Sequential
    """
    def forward(self, x):
        for module in self._modules.values(): #.values():
            # print(module)
            x = module(x)
        return x

    """
    Preprocess X and U (as they would be outputted by dynamics.generate_data) so they can be passed into the neural network for training
    X and U should be numpy arrays with dimensionality X.shape = (num_iter, sequence_len, 12) U.shape = (num_iter, sequence_len, 4)

    return: A
    """
    def preprocess(self, X, U):

        #Getting output dX
        dX = np.array([utils_data.states2delta(val) for val in X])


        # Ignore last element of X and U sequences because do not see next state
        X = X[:,:-1,:]
        U = U[:,:-1,:]

        #translating from [psi theta phi] to [sin(psi)  sin(theta) sin(phi) cos(psi) cos(theta) cos(phi)]
        # modX = np.concatenate((X[:, :, 0:6], np.sin(X[:, :, 6:9]), np.cos(X[:, :, 6:9]), X[:, :, 9:]), axis=2)
        # dX = np.concatenate((dX[:, :, 0:6], np.sin(dX[:, :, 6:9]), np.cos(dX[:, :, 6:9]), dX[:, :, 9:]), axis=2)

        # Adds the desired variables to the X data to learn
        modX = []
        moddX = []

        # Adds the desired inputs to the U data to learn X with
        modU = []
        for i in range(np.shape(X)[0]):
            seqX = X[i,:,:]
            seqdX = dX[i,:,:]
            seqU = U[i,:,:]

            # intialize empty arrays to make slicing easier
            arr_X = []
            arr_dX = []
            arr_U = []
            for state in self.state_learn_list:
                # grabs state information from dictionary
                key = self.dynam.x_dict[state]

                # concatenate required variables for states
                if (key[1] != 'angle'):
                    arr_X.append(seqX[:, key[0]])
                    arr_dX.append(seqdX[:, key[0]])
                else:
                    arr_X.append(np.sin(seqX[:, key[0]]) )
                    arr_X.append(np.cos(seqX[:, key[0]]) )
                    arr_dX.append(np.sin(seqdX[:, key[0]]))
                    arr_dX.append(np.cos(seqdX[:, key[0]]) )


            for inp in self.input_learn_list:

                # grabs state information from dictionary
                key = self.dynam.u_dict[inp]

                # concatenate required variables for states
                arr_U.append(seqU[:, key[0]])

            # append the slice onto the array
            modX.append(arr_X)
            moddX.append(arr_dX)
            modU.append(arr_U)

        # cast to numpy arrays
        modX = np.array(modX)
        moddX = np.array(moddX)
        modU = np.array(modU)

        # swap axes for easy flatten & tensor
        modX = np.swapaxes(modX, 1, 2)
        moddX = np.swapaxes(moddX, 1, 2)
        modU = np.swapaxes(modU, 1, 2)

        #Follow by flattening the matrices so they look like input/output pairs
        modX = modX.reshape(modX.shape[0]*modX.shape[1], -1)
        modU = modU.reshape(modU.shape[0]*modU.shape[1], -1)
        moddX = moddX.reshape(dX.shape[0]*dX.shape[1], -1)

        #at this point they should look like input output pairs
        if moddX.shape != modX.shape:
            raise ValueError('Something went wrong, modified X shape:' + str(modX.shape) + ' dX shape:' + str(dX.shape))

        #update mean and variance of the dataset with each training pass
        self.scalarX.partial_fit(modX)
        self.scalarU.partial_fit(modU)
        self.scalardX.partial_fit(moddX)

        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(modX)
        normU = self.scalarU.transform(modU)
        normdX = self.scalardX.transform(moddX)
        print(np.shape(normX))
        print(np.shape(normU))
        print(np.shape(normdX))

        # quit()
        # print(self.scalarX.mean_)
        # print(self.scalarU.mean_)
        # print(self.scalardX.mean_)
        # print(self.scalarX.var_)
        # print(self.scalarU.var_)
        # print(self.scalardX.var_)
        # print(self.scalarX.n_samples_seen_)
        # print(self.scalarU.n_samples_seen_)
        # print(self.scalardX.n_samples_seen_)
        # print(np.mean(normX))
        # print(np.mean(normU))
        # print(np.mean(normdX))

        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(normdX)

        return list(zip(inputs, outputs))



    """
    Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset and converting from cos, sin
    to actual angle
    """
    def postprocess(self, dX):
        # de-normalize so to say
        dX = self.scalardX.inverse_transform(dX.reshape(1, -1))
        dX = dX.ravel()
        # print(np.shape(dX))
        out = []
        ang_idx = 0

        def NNout2State(dX):
            # helper function for transforming the output of the NN back to useable state information

            out = []
            # Again needs to remove cos/sin of the correct variables the desired variables to the X data to learn

            l = 0
            # grabs state information from dictionary
            for (i,state) in enumerate(self.state_learn_list):
                # grabs state information from dictionary
                key = self.dynam.x_dict[state]

                # concatenate required variables for states
                if (key[1] != 'angle'):
                    out.append(dX[i+l])
                else:
                    # out.append(np.arctan2(dX[1+i+l], dX[i+l]))
                    out.append(np.arctan2(dX[i+l], dX[1+i+l]))
                    l+= 1
            return out

        # Call normalization on each state predicted
        if len(np.shape(dX)) > 1:
            for state in dX:
                out.append(NNout2State(state))
        else:
            out = NNout2State(dX)

        return np.array(out)
        # out = np.concatenate((dX[:, :6], np.arctan2(dX[:, 6:9], dX[:, 9:12]), dX[:, 12:]), axis=1)


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
            print('Shape of dataset is:', len(dataset))

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
        # state_in = np.concatenate((X[0:6], np.sin(X[6:9]), np.cos(X[6:9]), X[9:]))
        state_in = []
        input_in = []

        for state in self.state_learn_list:
            # grabs state information from dictionary
            key = self.dynam.x_dict[state]

            # concatenate required variables for states
            if (key[1] != 'angle'):
                state_in.append(X[key[0]])
            else:
                state_in.append(np.sin(X[key[0]]) )
                state_in.append(np.cos(X[key[0]]) )


        for inp in self.input_learn_list:

            # grabs state information from dictionary
            key = self.dynam.u_dict[inp]

            # concatenate required variables for states
            input_in.append(U[key[0]])

        # make numpy array
        state_in = np.array(state_in)
        input_in = np.array(input_in)

        #normalizing and converting to single sample
        normX = self.scalarX.transform(state_in.reshape(1, -1))
        normU = self.scalarU.transform(input_in.reshape(1, -1))

        input = Variable(torch.Tensor(np.concatenate((normX, normU), axis=1)))

        NNout = self.postprocess(self.forward(input).data[0])

        # need to make it so you can still simulate sequences on the learned sequences of not all variables. Our definition is for now that prediction is set to 0 for states we did not learn
        out = np.zeros(self.dynam.x_dim)
        idx_out = 0
        for state in self.dynam.x_dict:
            key = self.dynam.x_dict[state]
            if state in self.state_learn_list:
                out[key[0]] = NNout[idx_out]
                idx_out += 1


        # Debug
        # print(np.shape(X))
        # print(np.shape(U))
        # print(np.shape(out))
        # print(out)
        return out



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

class EnsembleNN(nn.Module):
    """
    Creates an ensembleNN of parameters optimized from the standard NN
    Dropout
    """
    def __init__(self, base_net, num_nets):
        # super(NeuralNet, self).__init__()

        #To keep track of what the mean and variance are at all times for transformations
        self.scalarX = StandardScaler()
        self.scalarU = StandardScaler()
        self.scalardX = StandardScaler()

        # list of states and inputs to learn dynamics from
        self.state_learn_list = base_net.state_learn_list
        self.input_learn_list = base_net.input_learn_list

        layer_sizes = [12, 100, 100, 9]
        layer_types = ['nn.Linear()','nn.ReLU()', 'nn.ReLU()', 'nn.Linear()']
        states_learn = ['yaw', 'pitch', 'roll', 'ax', 'ay', 'az']
        # ['X', 'Y', 'Z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll', 'w_z', 'w_x', 'w_y']
        forces_learn = ['Thrust', 'taux', 'tauy']

        # dynam file for reference
        self.dynam =  base_net.dynam

        # Creates list of nueral nets for processing
        self.num_nets = num_nets
        self.nets = []
        for i in range(num_nets):
            self.nets.append(copy.deepcopy(base_net))

    def forward_ens(self, x):
        """
        Standard forward function necessary if extending nn.Module. Basically a copy of nn.Sequential. Updated for bootstrap method to pass each net once
        """
        xs = []
        for net in self.nets:
            x_sub = x
            for module in net._modules.values():
                x_sub = module(x_sub)
            xs.append(x_sub)
        # print(np.shape(xs))
        x = torch.mean(torch.stack(xs), 0)
        return x

    def preprocess(self, X, U):
        """
        Preprocess X and U (as they would be outputted by dynamics.generate_data) so they can be passed into the neural network for training
        X and U should be numpy arrays with dimensionality X.shape = (num_iter, sequence_len, 12) U.shape = (num_iter, sequence_len, 4)

        return: A
        """

        #Getting output dX
        dX = np.array([utils_data.states2delta(val) for val in X])

        # Ignore last element of X and U sequences because do not see next state
        X = X[:,:-1,:]
        U = U[:,:-1,:]

        #translating from [psi theta phi] to [sin(psi)  sin(theta) sin(phi) cos(psi) cos(theta) cos(phi)]
        # modX = np.concatenate((X[:, :, 0:6], np.sin(X[:, :, 6:9]), np.cos(X[:, :, 6:9]), X[:, :, 9:]), axis=2)
        # dX = np.concatenate((dX[:, :, 0:6], np.sin(dX[:, :, 6:9]), np.cos(dX[:, :, 6:9]), dX[:, :, 9:]), axis=2)

        # Adds the desired variables to the X data to learn
        modX = []
        moddX = []

        # Adds the desired inputs to the U data to learn X with
        modU = []
        for i in range(np.shape(X)[0]):
            seqX = X[i,:,:]
            seqdX = dX[i,:,:]
            seqU = U[i,:,:]

            # intialize empty arrays to amke slicing easier
            arr_X = []
            arr_dX = []
            arr_U = []
            for state in self.state_learn_list:
                # grabs state information from dictionary
                key = self.dynam.x_dict[state]

                # concatenate required variables for states
                if (key[1] != 'angle'):
                    arr_X.append(seqX[:, key[0]])
                    arr_dX.append(seqdX[:, key[0]])
                else:
                    arr_X.append(np.sin(seqX[:, key[0]]) )
                    arr_X.append(np.cos(seqX[:, key[0]]) )
                    arr_dX.append(np.sin(seqdX[:, key[0]]))
                    arr_dX.append(np.cos(seqdX[:, key[0]]) )

            for inp in self.input_learn_list:

                # grabs state information from dictionary
                key = self.dynam.u_dict[inp]

                # concatenate required variables for states
                arr_U.append(seqU[:, key[0]])

            # append the slice onto the array
            modX.append(arr_X)
            moddX.append(arr_dX)
            modU.append(arr_U)

        # cast to numpy arrays
        modX = np.array(modX)
        moddX = np.array(moddX)
        modU = np.array(modU)

        # swap axes for easy flatten & tensor
        modX = np.swapaxes(modX, 1, 2)
        moddX = np.swapaxes(moddX, 1, 2)
        modU = np.swapaxes(modU, 1, 2)

        #Follow by flattening the matrices so they look like input/output pairs
        modX = modX.reshape(modX.shape[0]*modX.shape[1], -1)
        modU = modU.reshape(modU.shape[0]*modU.shape[1], -1)
        moddX = moddX.reshape(dX.shape[0]*dX.shape[1], -1)

        #at this point they should look like input output pairs
        if moddX.shape != modX.shape:
            raise ValueError('Something went wrong, modified X shape:' + str(modX.shape) + ' dX shape:' + str(dX.shape))

        #update mean and variance of the dataset with each training pass
        self.scalarX.partial_fit(modX)
        self.scalarU.partial_fit(modU)
        self.scalardX.partial_fit(moddX)

        #Normalizing to zero mean and unit variance
        normX = self.scalarX.transform(modX)
        normU = self.scalarU.transform(modU)
        normdX = self.scalardX.transform(moddX)

        inputs = torch.Tensor(np.concatenate((normX, normU), axis=1))
        outputs = torch.Tensor(normdX)

        # debugging
        print('Preprocessing sizes:')
        print(' ', np.shape(inputs))
        print(' ', np.shape(outputs))
        print('  ------')
        return list(zip(inputs, outputs))

    def postprocess(self, dX):
        """
        Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset and converting from cos, sin
        to actual angle
        """
        # de-normalize so to say
        dX = self.scalardX.inverse_transform(dX)

        out = []
        ang_idx = 0

        def NNout2State(dX):
            # helper function for transforming the output of the NN back to useable state information

            out = []
            # Again needs to remove cos/sin of the correct variables the desired variables to the X data to learn

            l = 0
            # grabs state information from dictionary
            for (i,state) in enumerate(self.state_learn_list):
                # grabs state information from dictionary
                key = self.dynam.x_dict[state]

                # concatenate required variables for states
                if (key[1] != 'angle'):
                    out.append(dX[i+l])
                else:
                    out.append(np.arctan2(dX[i+l], dX[1+i+l]))
                    l+= 1
            return out

        # Call normalization on each state predicted
        if len(np.shape(dX)) > 1:
            for state in dX:
                out.append(NNout2State(state))
        else:
            out = NNout2State(dX)

        return np.array(out)
        # out = np.concatenate((dX[:, :6], np.arctan2(dX[:, 6:9], dX[:, 9:12]), dX[:, 12:]), axis=1)

    def train_ens(self, dataset, learning_rate = 1e-3, epochs=50, batch_size=50, optim="Adam", loss_fn=nn.MSELoss(), split=0.9, preprocess=True):
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
            print('Length of dataset is:', len(dataset))


        num_samples = len(dataset)
        weights = (1/(num_samples+1))*np.ones((int(split*num_samples)))
        # TODO: Update datasets by sampling with replacement for each net

        # Make random sampling with replacement by using a evenly weighted random sampler with replacement
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

        # Training loader has the sampler, testing does not matter.
        trainLoader = DataLoader(dataset[:int(split*len(dataset))], sampler = sampler, batch_size=batch_size)
        testLoader = DataLoader(dataset[int(split*len(dataset)):], batch_size=batch_size)


        # TODO: Train each net separately
        #Unclear if we should be using SGD or ADAM? Papers seem to say ADAM works better

        # train each net
        errors = []
        for i, net in enumerate(self.nets):
            if(optim=="Adam"):
                optimizer = torch.optim.Adam(super(NeuralNet, net).parameters(), lr=learning_rate)
            elif(optim=="SGD"):
                optimizer = torch.optim.SGD(super(NeuralNet, net).parameters(), lr=learning_rate)
            else:
                raise ValueError(optim + " is not a valid optimizer type")
            print('Training net ', i+1)
            error = net._optimize(loss_fn, optimizer, epochs, batch_size, trainLoader, testLoader)
            errors.append(error)
            print('-------------------------------------------------------')
        print(np.shape(errors))
        return errors


    def predict(self, X, U):
        """
        Given a state X and input U, predict the change in state dX. This function does all pre and post processing for the neural net
        """
        #Converting to sin/cos form
        # state_in = np.concatenate((X[0:6], np.sin(X[6:9]), np.cos(X[6:9]), X[9:]))

        state_in = []
        input_in = []

        for state in self.state_learn_list:
            # grabs state information from dictionary
            key = self.dynam.x_dict[state]

            # concatenate required variables for states
            if (key[1] != 'angle'):
                state_in.append(X[key[0]])
            else:
                state_in.append(np.sin(X[key[0]]) )
                state_in.append(np.cos(X[key[0]]) )


        for inp in self.input_learn_list:

            # grabs state information from dictionary
            key = self.dynam.u_dict[inp]

            # concatenate required variables for states
            input_in.append(U[key[0]])

        # make numpy array
        state_in = np.array(state_in)
        input_in = np.array(input_in)

        #normalizing and converting to single sample
        normX = self.scalarX.transform(state_in.reshape(1, -1))
        normU = self.scalarU.transform(input_in.reshape(1, -1))

        input = Variable(torch.Tensor(np.concatenate((normX, normU), axis=1)))

        NNout = self.postprocess(self.forward_ens(input).data[0])

        # need to make it so you can still simulate sequences on the learned sequences of not all variables. Our definition is for now that prediction is set to 0 for states we did not learn
        out = np.zeros(self.dynam.x_dim)
        idx_out = 0
        for state in self.dynam.x_dict:
            key = self.dynam.x_dict[state]
            if state in self.state_learn_list:
                out[key[0]] = NNout[idx_out]
                idx_out += 1
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

    # TODO: Reimplement with pickle, getting issues with torch's built in with the save/load operations inheriting the wrong .predict() function on a loaded model. Running a model which was trained in loop worked.
    def save_model(self, filepath):
        # torch.save(self.state_dict(), filepath)   # only param
        torch.save(self, filepath)                  # full

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
