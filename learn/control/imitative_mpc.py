from utils.data import *
from utils.sim import *
from utils.nn import *
from utils.rl import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN
# from model_split_nn import SplitModel
# from model_ensemble_nn import EnsembleNN

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib


def generate_mpc_imitate(dataset, data_params, nn_params, train_params):
    """
    Will be used for imitative control of the model predictive controller. 
    Could try adding noise to the sampled acitons...
    """

    class ImitativePolicy(nn.Module):
        def __init__(self, nn_params):
            super(ImitativePolicy, self).__init__()

            # Store the parameters:
            self.hidden_w = nn_params['hid_width']
            self.depth = nn_params['hid_depth']

            self.n_in_input = nn_params['dx']
            self.n_out = nn_params['du']

            self.activation = nn_params['activation']
            self.d = nn_params['dropout']

            self.loss_fnc = nn.MSELoss()

            # super(ImitativePolicy, self).__init__()

            # Takes objects from the training parameters
            layers = []
            layers.append(nn.Linear(self.n_in_input, self.hidden_w)
                          )       # input layer
            layers.append(self.activation)
            layers.append(nn.Dropout(p=self.d))
            for d in range(self.depth):
                # add modules
                # input layer
                layers.append(nn.Linear(self.hidden_w, self.hidden_w))
                layers.append(self.activation)
                layers.append(nn.Dropout(p=self.d))

            # output layer
            layers.append(nn.Linear(self.hidden_w, self.n_out))
            self.features = nn.Sequential(*layers)

            # Need to scale the state variables again etc
            # inputs state, output an action (PWMs)
            self.scalarX = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))
            self.scalarU = MinMaxScaler(feature_range=(-1, 1))

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = self.features(x)

            return x

        def preprocess(self, dataset):  # X, U):
            """
            Preprocess X and U for passing into the neural network. For simplicity, takes in X and U as they are output from generate data, but only passed the dimensions we want to prepare for real testing. This removes a lot of potential questions that were bugging me in the general implementation. Will do the cosine and sin conversions externally.
            """
            # Already done is the transformation from
            # [yaw, pitch, roll, x_ddot, y_ddot, z_ddot]  to
            # [sin(yaw), sin(pitch), sin(roll), cos(pitch), cos(yaw),  cos(roll), x_ddot, y_ddot, z_ddot]
            # dX = np.array([utils_data.states2delta(val) for val in X])
            if len(dataset) == 2:
                X = dataset[0]
                U = dataset[1]
            else:
                raise ValueError("Improper data shape for training")

            self.scalarX.fit(X)
            self.scalarU.fit(U)

            #Normalizing to zero mean and unit variance
            normX = self.scalarX.transform(X)
            normU = self.scalarU.transform(U)

            inputs = torch.Tensor(normX)
            outputs = torch.Tensor(normU)

            return list(zip(inputs, outputs))

        def postprocess(self, U):
            """
            Given the raw output from the neural network, post process it by rescaling by the mean and variance of the dataset
            """
            # de-normalize so to say
            U = self.U.inverse_transform(U.reshape(1, -1))
            U = U.ravel()
            return np.array(U)

        def train_cust(self, dataset, train_params, gradoff=False):
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
            epochs = train_params['epochs']
            batch_size = train_params['batch_size']
            optim = train_params['optim']
            split = train_params['split']
            lr = train_params['lr']
            lr_step_eps = train_params['lr_schedule'][0]
            lr_step_ratio = train_params['lr_schedule'][1]
            preprocess = train_params['preprocess']

            if preprocess:
                dataset = self.preprocess(dataset)  # [0], dataset[1])

            trainLoader = DataLoader(
                dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)
            testLoader = DataLoader(
                dataset[int(split*len(dataset)):], batch_size=batch_size)

            # Papers seem to say ADAM works better
            if(optim == "Adam"):
                optimizer = torch.optim.Adam(
                    super(ImitativePolicy, self).parameters(), lr=lr)
            elif(optim == "SGD"):
                optimizer = torch.optim.SGD(
                    super(ImitativePolicy, self).parameters(), lr=lr)
            else:
                raise ValueError(optim + " is not a valid optimizer type")

            # most results at .6 gamma, tried .33 when got NaN
            if lr_step_eps != []:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=lr_step_eps, gamma=lr_step_ratio)

            testloss, trainloss = self._optimize(
                self.loss_fnc, optimizer, split, scheduler, epochs, batch_size, dataset)  # trainLoader, testLoader)

            return testloss, trainloss

        def predict(self, X):
            """
            Given a state X, predict the desired action U. This function is used when simulating, so it does all pre and post processing for the neural net
            """

            #normalizing and converting to single sample
            normX = self.scalarX.transform(X.reshape(1, -1))

            input = torch.Tensor(normX)

            NNout = self.forward(input).data[0]

            return NNout

        # trainLoader, testLoader):
        def _optimize(self, loss_fn, optim, split, scheduler, epochs, batch_size, dataset, gradoff=False):
            errors = []
            error_train = []
            split = split

            testLoader = DataLoader(
                dataset[int(split*len(dataset)):], batch_size=batch_size)
            trainLoader = DataLoader(
                dataset[:int(split*len(dataset))], batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                scheduler.step()
                avg_loss = torch.zeros(1)
                num_batches = len(trainLoader)/batch_size
                for i, (input, target) in enumerate(trainLoader):
                    # Add noise to the batch
                    if False:
                        if self.prob:
                            n_out = int(self.n_out/2)
                        else:
                            n_out = self.n_out
                        noise_in = torch.tensor(np.random.normal(
                            0, .01, (input.size())), dtype=torch.float)
                        noise_targ = torch.tensor(np.random.normal(
                            0, .01, (target.size())), dtype=torch.float)
                        input.add_(noise_in)
                        target.add_(noise_targ)

                    optim.zero_grad()                             # zero the gradient buffers
                    # compute the output
                    output = self.forward(input)

                    loss = loss_fn(output, target)
                    # add small loss term on the max and min logvariance if probablistic network
                    # note, adding this term will backprob the values properly

                    if loss.data.numpy() == loss.data.numpy():
                        # print(self.max_logvar, self.min_logvar)
                        if not gradoff:
                            # backpropagate from the loss to fill the gradient buffers
                            loss.backward()
                            optim.step()                                  # do a gradient descent step
                        # print('tain: ', loss.item())
                    # if not loss.data.numpy() == loss.data.numpy(): # Some errors make the loss NaN. this is a problem.
                    else:
                        # This is helpful: it'll catch that when it happens,
                        print("loss is NaN")
                        # print("Output: ", output, "\nInput: ", input, "\nLoss: ", loss)
                        errors.append(np.nan)
                        error_train.append(np.nan)
                        # and give the output and input that made the loss NaN
                        return errors, error_train
                    # update the overall average loss with this batch's loss
                    avg_loss += loss.item()/(len(trainLoader)*batch_size)

                # self.features.eval()
                test_error = torch.zeros(1)
                for i, (input, target) in enumerate(testLoader):

                    output = self.forward(input)
                    loss = loss_fn(output, target)

                    test_error += loss.item()/(len(testLoader)*batch_size)
                test_error = test_error

                #print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]),
                #          "test_error={:.9f}".format(test_error))
                if (epoch % 1 == 0):
                    print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(
                        avg_loss.data[0]), "test loss=", "{:.6f}".format(test_error.data[0]))
                # if (epoch % 50 == 0) & self.prob: print(self.max_logvar, self.min_logvar)
                error_train.append(avg_loss.data[0].numpy())
                errors.append(test_error.data[0].numpy())
            #loss_fn.print_mmlogvars()
            return errors, error_train

    # create policy object
    policy = ImitativePolicy(nn_params)

    # train policy
    # X, U, _ = df_to_training(df, data_params)
    X = dataset[0]
    U = dataset[1]
    acctest, acctrain = policy.train_cust((X, U), train_params)

    if True:
        ax1 = plt.subplot(211)
        # ax1.set_yscale('log')
        ax1.plot(acctest, label='Test Loss')
        plt.title('Test Loss')
        ax2 = plt.subplot(212)
        # ax2.set_yscale('log')
        ax2.plot(acctrain, label='Train Loss')
        plt.title('Training Loss')
        ax1.legend()
        plt.show()

    # return policy!
    return policy

if __name__ == '__main__':
    raise NotImplementedError("Need to add policy functions to code base")
    model = TanhGaussianPolicy(
        hidden_sizes=[300, 300],
        obs_dim=35,
        action_dim=4,
    )
    model.load_state_dict(torch.load('_policies/test.pth'))
    model.eval()
    model.training = False
    print(model.forward(torch.zeros(35)))
    quit()

    load_params = {
        'delta_state': True,                # normally leave as True, prediction mode
        # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
        'include_tplus1': True,
        # trims high vbat because these points the quad is not moving
        'trim_high_vbat': 4050,
        # If not trimming data with fast log, need another way to get rid of repeated 0s
        'takeoff_points': 180,
        # if all the euler angles (floats) don't change, it is not realistic data
        'trim_0_dX': True,
        'find_move': True,
        # if the states change by a large amount, not realistic
        'trime_large_dX': True,
        # Anything out of here is erroneous anyways. Can be used to focus training
        'bound_inputs': [20000, 65500],
        # IMPORTANT ONE: stacks the past states and inputs to pass into network
        'stack_states': 3,
        # looks for sharp changes to tthrow out items post collision
        'collision_flag': False,
        # shuffle pre training, makes it hard to plot trajectories
        'shuffle_here': False,
        'timestep_flags': [],               # if you want to filter rostime stamps, do it here
        'battery': True,                   # if battery voltage is in the state data
        # adds a column to the dataframe tracking end of trajectories
        'terminals': True,
        'fastLog': True,                   # if using the software with the new fast log
        # Number of times the control freq you will be using is faster than that at data logging
        'contFreq': 1,
        'iono_data': True,
        'zero_yaw': True,
        'moving_avg': 7
    }


    # dir_list = ["_newquad1/publ2/c50_rand/",
    #             "_newquad1/publ2/c50_roll01/",
    #             "_newquad1/publ2/c50_roll02/",
    #             "_newquad1/publ2/c50_roll03/",
    #             "_newquad1/publ2/c50_roll04/",
    #             "_newquad1/publ2/c50_roll05/",
    #             "_newquad1/publ2/c50_roll06/",
    #             "_newquad1/publ2/c50_roll07/",
    #             "_newquad1/publ2/c50_roll08/",
    #             "_newquad1/publ2/c50_roll09/",
    #             "_newquad1/publ2/c50_roll10/",
    #             "_newquad1/publ2/c50_roll11/",
    #             "_newquad1/publ2/c50_roll12/"]

    dir_list = ["_newquad1/publ2/c25_roll08/",
                "_newquad1/publ2/c25_roll09/",
                "_newquad1/publ2/c25_roll10/",
                "_newquad1/publ2/c25_roll11/",
                "_newquad1/publ2/c25_roll12/"]


    # quit()
    df = load_dirs(dir_list, load_params)


    data_params = {
        # Note the order of these matters. that is the order your array will be in
        'states': ['omega_x0', 'omega_y0', 'omega_z0',
                'pitch0',   'roll0',    'yaw0',
                'lina_x0',  'lina_y0',  'lina_z0',
                'omega_x1', 'omega_y1', 'omega_z1',
                'pitch1',   'roll1',    'yaw1',
                'lina_x1',  'lina_y1',  'lina_z1',
                'omega_x2', 'omega_y2', 'omega_z2',
                'pitch2',   'roll2',    'yaw2',
                'lina_x2',  'lina_y2',  'lina_z2'],
        # 'omega_x3', 'omega_y3', 'omega_z3',
        # 'pitch3',   'roll3',    'yaw3',
        # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs': ['m1pwm_0', 'm2pwm_0', 'm3pwm_0', 'm4pwm_0',
                'm1pwm_1', 'm2pwm_1', 'm3pwm_1', 'm4pwm_1',
                'm1pwm_2', 'm2pwm_2', 'm3pwm_2', 'm4pwm_2'],  # 'vbat'],
        # 'm1pwm_3', 'm2pwm_3', 'm3pwm_3', 'm4pwm_3', 'vbat'],

        'targets': ['t1_omegax', 't1_omegay', 't1_omegaz',
                    'd_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],

        'battery': False                    # Need to include battery here too
    }

    # the true state target values
    # 't1_omegax', 't1_omegay', 't1_omegaz', 't1_pitch', 't1_roll', 't1_yaw', 't1_linax', 't1_linay' 't1_linaz'

    st = ['d_omegax', 'd_omegay', 'd_omegaz',
        'd_pitch', 'd_omegaz', 'd_pitch',
        'd_linax', 'd_linay', 'd_linyz']

    X, U, dX = df_to_training(df, data_params)


    nn_params = {                           # all should be pretty self-explanatory
        'dx': np.shape(X)[1],
        'du': np.shape(U)[1],
        'dt': np.shape(dX)[1],
        'hid_width': 50,
        'hid_depth': 2,
        'bayesian_flag': True,
        'activation': Swish(),
        'dropout': 0.1,
        'split_flag': False,
        'pred_mode': 'Delta State',
        'ensemble': False
    }

    train_params = {
        'epochs': 45,
        'batch_size': 18,
        'optim': 'Adam',
        'split': 0.8,
        'lr': .0003,  # bayesian .00175, mse:  .0001
        'lr_schedule': [30, .6],
        'test_loss_fnc': [],
        'preprocess': True,
        'noprint': True
    }

    policy = generate_mpc_imitate((X,U,dX), data_params, nn_params, train_params)

