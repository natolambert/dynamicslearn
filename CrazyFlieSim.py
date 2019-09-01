'''Code from: Nathan Lambert'''

import numpy as np
import math
from PIDPolicy import policy
from Initial import getState, degToRad, radToDeg
from Visualizer import GUI
import matplotlib.pyplot as plt
from RolloutAnalysis import Analysis

class CrazyFlie():
    def __init__(self, dt, m=.035, L=.065, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, x_noise=.001, u_noise=0):
        self._state_dict = {
            'X': [0, 'pos'],
            'Y': [1, 'pos'],
            'Z': [2, 'pos'],
            'vx': [3, 'vel'],
            'vy': [4, 'vel'],
            'vz': [5, 'vel'],
            'yaw': [6, 'angle'],
            'pitch': [7, 'angle'],
            'roll': [8, 'angle'],
            'w_x': [9, 'omega'],
            'w_y': [10, 'omega'],
            'w_z': [11, 'omega']
        }

        self._input_dict = {
            'Thrust': [0, 'force'],
            'taux': [1, 'torque'],
            'tauy': [2, 'torque'],
            'tauz': [3, 'torque']
        }
        self.x_dim =12
        self.u_dim = 4
        self.dt = dt
        self.x_noise = x_noise

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        # Setup the parameters
        self.m = m
        self.L = L
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = -9.81

        # Define equilibrium input for quadrotor around hover
        self.u_e = np.array([m*self.g, 0, 0, 0])               #This is not the case for PWM inputs
        # Four PWM inputs around hover, extracted from mean of clean_hover_data.csv
        # self.u_e = np.array([42646, 40844, 47351, 40116])

        # Hover control matrices
        self._hover_mats = [np.array([1, 0, 0, 0]),      # z
                            np.array([0, 1, 0, 0]),   # pitch
                            np.array([0, 0, 1, 0])]   # roll
        #variable to keep track of most recent policy test data
        self.recentDataset = None

    def pqr2rpy(self, x0, pqr):
        #x0: yaw/psi, pitch/theta, roll/phi
        rotn_matrix = np.array([[1., math.sin(x0[2]) * math.tan(x0[1]), math.cos(x0[2]) * math.tan(x0[1])],
                                [0., math.cos(x0[2]),-math.sin(x0[2])],
                                [0., math.sin(x0[2]) / math.cos(x0[1]), math.cos(x0[2]) / math.cos(x0[1])]])
        #x02 is the roll angle and x01 is the pitch angle...x00 would be the yaw angle
        #pqr should be rotations around x,y,then z from body frame of the sensor (pitch rate, roll rate, yaw rate)
        return np.flip(rotn_matrix.dot(pqr), 0)

    def pwm_thrust_torque(self, PWM):
        # Takes in the a 4 dimensional PWM vector and returns a vector of
        # [Thrust, Taux, Tauy, Tauz] which is used for simulating rigid body dynam
        # Sources of the fit: https://wiki.bitcraze.io/misc:investigations:thrust,
        #   http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8905295&fileOId=8905299

        # The quadrotor is 92x92x29 mm (motor to motor, square along with the built in prongs). The the distance from the centerline,

        # Thrust T = .35*d + .26*d^2 kg m/s^2 (d = PWM/65535 - normalized PWM)
        # T = (.409e-3*pwm^2 + 140.5e-3*pwm - .099)*9.81/1000 (pwm in 0,255)

        def pwm_to_thrust(PWM):
            # returns thrust from PWM
            pwm_n = PWM/65535.0
            thrust = .35*pwm_n + .26*pwm_n**2
            return thrust

        pwm_n = np.sum(PWM)/(4*65535.0)

        l = 35.527e-3   # length to motors / axis of rotation for xy
        lz = 46         # axis for tauz
        c = .05         # coupling coefficient for yaw torque

        # Torques are slightly more tricky
        # x = m2+m3-m1-m4
        # y =m1+m2-m3-m4

        # Estiamtes forces
        m1 = pwm_to_thrust(PWM[0])
        m2 = pwm_to_thrust(PWM[1])
        m3 = pwm_to_thrust(PWM[2])
        m4 = pwm_to_thrust(PWM[3])

        Thrust = pwm_to_thrust(np.sum(PWM)/(4*65535.0))
        taux = l*(m2+m3-m4-m1)
        tauy = l*(m1+m2-m3-m4)
        tauz = -lz*c*(m1+m3-m2-m4)

        return np.array([Thrust, taux, tauy, tauz])

    def simulate(self, x, PWM, t=None, addNoise = False):
        # Input structure:
        # u1 = thrust
        # u2 = torque-wx
        # u3 = torque-wy
        # u4 = torque-wz
        u = self.pwm_thrust_torque(PWM)
        dt = self.dt
        u0 = u
        x0 = x
        idx_xyz = self.idx_xyz
        idx_xyz_dot = self.idx_xyz_dot
        idx_ptp = self.idx_ptp
        idx_ptp_dot = self.idx_ptp_dot

        m = self.m
        L = self.L
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        g = self.g

        Tx = np.array([Iyy / Ixx - Izz / Ixx, L / Ixx])
        Ty = np.array([Izz / Iyy - Ixx / Iyy, L / Iyy])
        Tz = np.array([Ixx / Izz - Iyy / Izz, 1. / Izz])

        # Array containing the forces
        Fxyz = np.zeros(3)
        Fxyz[0] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.cos(
            x0[idx_ptp[2]]) + math.sin(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[1] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.sin(
            x0[idx_ptp[2]]) - math.sin(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[2] = g + 1 * (math.cos(x0[idx_ptp[0]]) *
                           math.cos(x0[idx_ptp[1]])) * u0[0] / m
        # Compute the torques
        t0 = np.array([x0[idx_ptp_dot[1]] * x0[idx_ptp_dot[2]], u0[1]])
        t1 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[2]], u0[2]])
        t2 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[1]], u0[3]])
        Txyz = np.array([Tx.dot(t0), Ty.dot(t1), Tz.dot(t2)])

        x1 = np.zeros(12)
        x1[idx_xyz_dot] = x0[idx_xyz_dot] + dt * Fxyz
        x1[idx_ptp_dot] = x0[idx_ptp_dot] + dt * Txyz
        x1[idx_xyz] = x0[idx_xyz] + dt * x0[idx_xyz_dot]
        x1[idx_ptp] = x0[idx_ptp] + dt * self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        if addNoise:
            x_noise_vec = np.random.normal(loc=0, scale=self.x_noise, size=(self.x_dim))
        # makes states less than 1e-12 = 0
            x1[x1 < 1e-12] = 0
            return x1+x_noise_vec
        else:
            x1[x1 < 1e-12]
            return x1

    def test_policy(self, simDict):

        visuals = simDict['visuals']
        initStates = simDict['initStates']
        iterations = simDict['simFlights']
        maxCond = simDict['maxAngle']
        maxFrames = simDict['maxRollout']
        addNoise = simDict['addNoise']
        policy = simDict['policy']
        if visuals:
            visualizer = GUI(1) #initializes graphic display of quadcopter if requested
        time = []
        firstAddToRecent = True
        i = 0
        avgError = 0
        pitchidx = self._state_dict['pitch'][0]
        rollidx = self._state_dict['roll'][0]
        yawidx = self._state_dict['yaw'][0]
        while i < iterations:
            index = np.random.randint(0, np.shape(initStates)[0]) #randomly find initial condition
            state = (initStates[index, :])
            if abs(state[pitchidx])> math.radians(10) or abs(state[rollidx])> math.radians(10) or abs(state[yawidx]) > math.radians(15):
                continue #find new input
            #print("")
            #print("Beginning flight number ", i + 1)
            frames = 0
            failed = False
            X0_temp = None
            U_temp = None
            X1_temp = None
            first = True
            error = 0
            max = math.radians(maxCond)

            pitch = []
            roll = []
            yaw = []
            while not failed and frames < maxFrames:
                #update the policy based on the DEGREES. (Returned states from simulation are in radians)
                policy.update([math.degrees(state[pitchidx]), math.degrees(state[rollidx]), math.degrees(state[yawidx])])
                PWM = policy.chooseAction() #choose random action and simulate based on state
                newState = self.simulate(state, PWM.numpy(), addNoise = addNoise)
                if visuals: #update the visualization
                    pos = newState[0:3]
                    euler = np.array([newState[pitchidx], newState[rollidx], newState[yawidx]])
                    visualizer.update(self.dt, pos, euler)
                if first:
                    X0_temp = state.reshape(1,-1)
                    U_temp = PWM.numpy().reshape(1,-1)
                    X1_temp = newState.reshape(1,-1)
                    first = False
                else: #store the results from the simulation
                    X0_temp = np.vstack((X0_temp, state))
                    U_temp = np.vstack((U_temp, PWM.numpy()))
                    X1_temp = np.vstack((X1_temp, newState))
                state = newState #prepare for next iteration
                frames += 1
                error += newState[pitchidx] + newState[rollidx] + newState[yawidx] #take account the error of this state
                if abs(state[pitchidx]) > max or abs(state[rollidx]) > max or abs(state[yawidx]) > max:
                    failed = True
            error = error / frames #avg error
            time += [frames]
            rewards = [[frames] for k in range(frames)] #keeps track of total flight time
            rewards = (np.array(rewards))
            flightnum = [[i] for j in range(frames)] #states which flight a datapoint is a part of
            flightnum = np.array(flightnum)
            result = np.hstack((X0_temp, U_temp, X1_temp, rewards, flightnum))
            i += 1
            if firstAddToRecent:
                self.recentDataSet = result
                firstAddToRecent = False
            else:
                self.recentDataSet = np.vstack((self.recentDataSet, result))
            avgError += (1/iterations) * error

        return sum(time), sum(time)/iterations, np.var(time), avgError

    def get_recent_data(self):
        return self.recentDataSet
