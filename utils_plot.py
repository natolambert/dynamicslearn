import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d
from model_general_nn import predict_nn


def plot_trajectories_state(Seqs_X, dim):
    '''
    Takes in sequences and plots the trajectories along a given state over time. This is used for debugging data generation now, and in the future for visualizing the predicted states.
    '''
    n, l, dx = np.shape(Seqs_X)

    data = Seqs_X[:,:,dim]

    # Plot
    modelacc_fig = plt.figure()
    title = 'Comparing Ground Truth of Dynamics to Model, Dim:' + str(dim)
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('State Value')
    [plt.plot(datum) for datum in data]
    # plt.plot(data, label='Generated Data')
    # plt.legend()
    plt.show()
    return modelacc_fig


def plot_model(data, model, dim, model_dims = [6,7,8,12,13,14], delta = True, sort=False):
    '''
    Function that takes in data of the form (states, inputs) and plots a specific state variable's ground truth and it's one step prediction. Ground truth is the actual state that occured and prediction is f(x,u) from the learned model. Dimension is the dimension of the state to look at, for simplicity.

    note this would work for 3-tuples of (next state, cur state, input) as well
    '''

    # Data of the form (dx, x, u) where dx, x, and u are sub arrays. Shape of data is (n_sample, 3).
    # data = sequencesXU2array(Seqs_X[:,::samp,:], Seqs_U[:,::samp,:]) from raw data
    # dxs = data[:,0]
    # xs = data[:,1]
    # us = data[:,2]
    #
    # if not delta:       # if trained for predicting raw state
    #     dxs = xs[:]
    #     # xs = xs[:-1]
    #     # us = us[:-1]
    #
    # # make data into matrices for a little bit easier unpacking
    # dxs = np.vstack(dxs)
    # xs = np.vstack(xs)
    # us = np.vstack(us)

    Seqs_X = data[0]
    Seqs_U = data[1]

    dxs = Seqs_X[1:,:]-Seqs_X[:-1,:]
    xs = Seqs_X[:-1,:]
    us = Seqs_U[:-1,:]

    # print(np.shape(dxs))
    # print(np.shape(xs))

    # Now need to iterate through all data and plot
    predictions = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(model,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions = np.append(predictions, pred.reshape(1,-1),  axis=0)

    # Debug
    # print(np.shape(predictions))

    # Grab correction dimension data
    ground_dim = dxs[:, dim]
    pred_dim = predictions[:, dim]

    # Sort with respect to ground truth
    if sort:
      ground_dim_sort, pred_dim_sort = zip(*sorted(zip(ground_dim,pred_dim)))
    else:
      ground_dim_sort, pred_dim_sort = ground_dim,pred_dim

    # Plot
    modelacc_fig = plt.figure()
    title = 'Comparing Ground Truth of Dynamics to Model, Dim:' + str(dim)
    plt.title(title)
    plt.xlabel('Sorted ground truth state index')
    plt.ylabel('State Value')
    plt.plot(pred_dim_sort, label='Predicted Val')
    plt.plot(ground_dim_sort, label='Ground Truth')
    plt.legend()
    #plt.show()
    return modelacc_fig
