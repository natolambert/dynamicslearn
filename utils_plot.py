import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d

def plot_trajectory(X,T):
    '''
    plots a trajectory given a list of state vectors
    plots trajectory over time
    '''
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
    '''
    Plots all state variables over time. Use this to debug state variables and dynamics files. ONLY WORKS FOR 12 DIM FREE BODY DYNAMICS
    '''
    if (np.shape(X)[1] != 12):
        raise ValueError('X dimension does not match, not equal 12')

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
    # plt.show()
    return states_fig

def plotInputs(U,T):
    '''
    Plots all control variables over time. Use this to visualize the forces from a quadrotor or ionocraft. THIS DOES NOT WORK FOR OTHER ROBOTS.
    '''
    if (np.shape(U)[1] != 4):
        raise ValueError('U dimension does not match, not equal 4')


    titles = ['U1', 'U2', 'U3', 'U4']
    y_lab = ['F (N)', 'F (N)', 'F (N)', 'F (N)']
    inputs_fig = plt.figure()
    for i in range(4):
        ax = plt.subplot(4,1,i+1)
        ax.set_title(titles[i])
        ax.set_xlabel('time (s)')
        ax.set_ylabel(y_lab[i])
        plt.plot(T,U[:,i])
    plt.tight_layout(w_pad = -1.5, h_pad= -.5)
    plt.show()
    return inputs_fig

def printState(x):
    '''
    Prints out the states with what they are, so it does not get cluttered.
    '''
    print('var:\t CURRENT STATE')
    print('X: \t', x[0])
    print('Y: \t', x[1])
    print('Z: \t', x[2])
    print('vx: \t', x[3])
    print('vy: \t', x[4])
    print('vz: \t', x[5])
    print('yaw: \t', x[6])
    print('pitch: \t', x[7])
    print('roll: \t', x[8])
    print('wx: \t', x[9])
    print('wy: \t', x[10])
    print('wz: \t', x[11])
    print()

def compareTraj(Seq_U, x0, dynamics_true, dynamics_learned, show = False):
    '''
    Plots in 3d the learned and true dynamics to compare visualization.
    '''
    Xtrue = np.array([x0])
    Xlearn = np.array([x0])

    for u in Seq_U:
        # Simulate Update Steps
        Xtrue_next = np.array([dynamics_true.simulate(Xtrue[-1,:],u)])
        Xlearn_next = np.array([Xtrue[-1,:] + dynamics_learned.predict(Xtrue[-1,:], u)])


        # Append Data
        Xtrue = np.append(Xtrue, Xtrue_next, axis=0)
        Xlearn = np.append(Xlearn, Xlearn_next, axis=0)


    # Plot ################################
    fig_compare = plt.figure()

    # 3D
    ax = fig_compare.add_subplot(111, projection="3d")

    # plot settings
    plt.axis("equal")
    ax.set_title('True Vs Learned Dynamics - Visualization')

    # plot labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # plot limits + padding
    plt_limits = np.array([[Xtrue[:, 0].min(), Xtrue[:, 0].max()],
                           [Xtrue[:, 1].min(), Xtrue[:, 1].max()],
                           [Xtrue[:, 2].min(), Xtrue[:,2].max()]])


    for item in plt_limits:
        if abs(item[1] - item[0]) < 1:
            item[0] -= .25
            item[1] += .25

    ax.set_xlim3d(plt_limits[0])
    ax.set_ylim3d(plt_limits[1])
    ax.set_zlim3d(plt_limits[2])

    # plot_trajectory
    ax.plot(Xtrue[:,0], Xtrue[:,1], Xtrue[:,2], 'k-', label='True Dynamics' )
    ax.plot(Xlearn[:,0], Xlearn[:,1], Xlearn[:,2], 'r--', label='Learned Dynamics' )
    ax.legend()

    if show:
        plt.show()
    return fig_compare





class PlotFlight(object):
    '''
    PlotFlight class adapted from: https://github.com/nikhilkalige/quadrotor/blob/master/plotter.py
    '''
    def __init__(self, state, arm_length):
        state[:,[6, 8]] = state[:,[8, 6]]

        self.state = state
        self.length = len(state)
        self.arm_length = arm_length

    def setup_plot(self):
        # setup
        self.fig = plt.figure()

        # Close button
        # self.fig.canvas.mpl_connect('close_event', self.fig.close())
        # plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
        #
        # def close(self):
        #     self.cap.stopPic()
        #     self.cap.close()
        #     self.cap.cleanCam()

        # 3D
        self.ax = self.fig.add_subplot(111, projection="3d")

        # plot settings
        plt.axis("equal")

        # plot labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # plot limits + padding
        plt_limits = np.array([[self.state[:, 0].min(), self.state[:, 0].max()],
                               [self.state[:, 1].min(), self.state[:, 1].max()],
                               [self.state[:, 2].min(), self.state[:,2].max()]])
        for item in plt_limits:
            if abs(item[1] - item[0]) < 2:
                item[0] -= 1
                item[1] += 1

        self.ax.set_xlim3d(plt_limits[0])
        self.ax.set_ylim3d(plt_limits[1])
        self.ax.set_zlim3d(plt_limits[2])

        # # # Hardcode axes instead
        # self.ax.set_xlim3d([-5, 5])
        # self.ax.set_ylim3d([-5, 5])
        # self.ax.set_zlim3d([-5, 5])

        # initialize the plot
        flight_path, = self.ax.plot([], [], [], '--')
        colors = ['r', 'g', 'b', 'y']
        arms = [self.ax.plot([], [], [], c=colors[i], marker='^')[0] for i in range(4)]
        self.plot_artists = [flight_path, arms]

    def rotate(self, euler_angles, point):
        [phi, theta, psi] = euler_angles
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        m = np.array([[cthe * cpsi, sphi * sthe * cpsi - cphi * spsi, cphi * sthe * cpsi + sphi * spsi],
                      [cthe * spsi, sphi * sthe * spsi + cphi * cpsi, cphi * sthe * spsi - sphi * cpsi],
                      [-sthe,       cthe * sphi,                      cthe * cphi]])

        return np.dot(m, point)

    def init_animate(self):
        self.plot_artists[0].set_data([], [])
        self.plot_artists[0].set_3d_properties([])

        for arm in self.plot_artists[1]:
            arm.set_data([], [])
            arm.set_3d_properties([])

        return [self.plot_artists[0]] + self.plot_artists[1]

    def animate(self, i):
        i = (i + 1) % (self.length + 1)
        x = self.state[:, 0][:i]
        y = self.state[:, 1][:i]
        z = self.state[:, 2][:i]

        center_point = np.array([x[-1], y[-1], z[-1]])
        euler_angles = self.state[i - 1][6:9]

        self.plot_artists[0].set_data(x, y)
        self.plot_artists[0].set_3d_properties(z)

        arm_base_pos = np.array([[self.arm_length, 0, 0],
                                 [0, -self.arm_length, 0],
                                 [-self.arm_length, 0, 0],
                                 [0, self.arm_length, 0]])

        arm_base_pos = [self.rotate(euler_angles, arm) for arm in arm_base_pos]

        # update the position
        arm_base_pos = [(arm + center_point) for arm in arm_base_pos]
        self.plot_arms(center_point, arm_base_pos)
        return [self.plot_artists[0]] + self.plot_artists[1]

    def plot_arms(self, center, arm_pos):
        arm_lines = self.plot_artists[1]
        for index, arm in enumerate(arm_pos):
            pos = np.column_stack((center, arm))
            arm_lines[index].set_data(pos[:2])
            arm_lines[index].set_3d_properties(pos[-1:])

    def show(self, save=False):
        self.setup_plot()
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init_animate,
                                       frames=self.length, interval=1,
                                       blit=True)
        plt.gca().set_aspect("equal", adjustable="box")
        if save:
            anim.save('Flight Path Anim.gif', writer='imagemagick', fps=30)
        plt.show()
