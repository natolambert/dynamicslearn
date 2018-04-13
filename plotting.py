import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(X,T):
    # plots a trajectory given a list of state vectors
    # plots a flying X with a colored axis representing the body frame on it
    # plots trajectory over time
    plot_size = 10
    iono_size = 1

    exfig = plt.figure()
    ax = exfig.add_subplot(111,projection = '3d')

    # plot the trajectory line in xyz
    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    yaws = X[:,6]
    pitches = X[:,7]
    rolls = X[:,8]

    ax.plot(xs,ys,zs=zs)

    # Axis limits
    ax.set_xlim([-.5,.5])
    ax.set_ylim([-.5,.5])
    ax.set_zlim([-.5,.5])
    ax.set_title('simulated single trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    plt.show()

def plot12(X,T):
    # Plots all state variables over time
    titles = ['X', 'Y', 'Z', 'xdot', 'ydot', 'zdot', 'yaw', 'pitch', 'roll', 'omega_x', 'omega_y', 'omega_z']
    y_lab = ['(m)', '(m)', '(m)', '(m/s)', '(m/s)', '(m/s)', '(rad)', '(rad)', '(rad)', '(rad/s)', '(rad/s)', '(rad/s)']
    states_fig = plt.figure()
    for i in range(12):
        ax = plt.subplot(4,3,i+1)
        ax.set_title(titles[i])
        ax.set_xlabel('time (s)')
        ax.set_ylabel(y_lab[i])
        plt.plot(T,X[:,i])
    plt.tight_layout(w_pad = -1.5, h_pad= -.5)
    plt.show()
