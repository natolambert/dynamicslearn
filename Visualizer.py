'''Visualizer for quadcopter. Runs no policy directly. Only a GUI for the user to analyze results.

All quadcopter states are computed elsewhere (for example in CrazyFlieSim.py)'''

import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys

class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, slow):
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-1.0, 1.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 20])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        self.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        self.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)
        self.slow = slow #parameter to tell by how many factors slower the simulation will run from real time
        print("The simulation will run ", slow," times slower than real time!")

    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


    def update(self, dt, position, euler):
        #position: x, y, z  euler: pitch, roll, yaw
        #updates the graphics to reflect new positions, orientations, and velocities
        R = self.rotation_matrix(euler)
        L = 1
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
        points = np.dot(R,points)
        points[0,:] += position[0]
        points[1,:] += position[1]
        points[2,:] += position[2]
        self.l1.set_data(points[0,0:2],points[1,0:2])
        self.l1.set_3d_properties(points[2,0:2])
        self.l2.set_data(points[0,2:4],points[1,2:4])
        self.l2.set_3d_properties(points[2,2:4])
        self.hub.set_data(points[0,5],points[1,5])
        self.hub.set_3d_properties(points[2,5])
        plt.pause(dt * self.slow)
