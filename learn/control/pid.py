from .controller import Controller

# timing etc
import time
import datetime
import os
import copy
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib


# basic PID class to perform arithmetic around the setpoint
class PID():
    def __init__(self, desired,
                 kp, ki, kd,
                 ilimit, dt, outlimit=np.inf,
                 samplingRate=0, cutoffFreq=-1,
                 enableDFilter=False):

        # internal variables
        self.error = 0
        self.error_prev = 0
        self.integral = 0
        self.deriv = 0
        self.out = 0

        # constants
        self.desired = desired
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # limits integral growth
        self.ilimit = ilimit

        # limits ridiculous actions. Should set to variance
        self.outlimit = outlimit

        # timee steps for changing step size of PID response
        self.dt = dt
        self.samplingRate = samplingRate  # sample rate is for filtering

        self.cutoffFreq = cutoffFreq
        self.enableDFilter = enableDFilter

        if cutoffFreq != -1 or enableDFilter:
            raise NotImplementedError('Have not implemnted filtering yet')

    def reset(self):
        # internal variables
        self.error = 0
        self.error_prev = 0
        self.integral = 0
        self.deriv = 0

    def update(self, measured):

        # init
        self.out = 0.

        # update error
        self.error_prev = self.error

        # calc new error
        self.error = self.desired - measured

        # proportional gain is easy
        self.out += self.kp * self.error

        # calculate deriv term
        self.deriv = (self.error - self.error_prev) / self.dt

        # filtter if needed (DT function_)
        if self.enableDFilter:
            print('Do Filter')
            self.deriv = self.deriv

        # calcualte error value added
        self.out += self.deriv * self.kd

        # accumualte normalized eerror
        self.integral = self.error * self.dt

        # limitt the integral term
        if self.ilimit != 0:
            self.integral = np.clip(self.integral, -self.ilimit, self.ilimit)

        self.out += self.ki * self.integral

        # limitt the total output
        if self.outlimit != 0:
            self.out = np.clip(self.out, -self.outlimit, self.outlimit)

        return self.out


'''
Some notes on the crazyflie PID structure. Essentially there is a trajectory planner
  that we can ignore, and a Attitude control that sents setpoints to a rate controller.
  The Attitude controller outputs a rate desired, and the rate desired updates motors

This is the code from the fimrware. You can see how the m1...m4 pwm values are set
  The motorPower.m1 is a pwm value, and limit thrust puts it in an range:
    motorPower.m1 = limitThrust(control->thrust + control->pitch +
                               control->yaw);
    motorPower.m2 = limitThrust(control->thrust - control->roll -
                               control->yaw);
    motorPower.m3 =  limitThrust(control->thrust - control->pitch +
                               control->yaw);
    motorPower.m4 =  limitThrust(control->thrust + control->roll -
                               control->yaw);

    This shows that m1 and m3 control pitch while m2 and m4 control roll.
    Yaw should account for a minor amount of this. Our setpoint will be easy,
    roll, pitch =0 ,yaw rate = 0.

Default values, for 250Hz control. Will expect our simulated values to differ:
Axis Mode: [KP, KI, KD, iLimit]

Pitch Rate: [250.0, 500.0, 2.5, 33.3]
Roll Rate: [250.0, 500.0, 2.5, 33.3]
Yaw Rate: [120.0, 16.7, 0.0, 166.7]
Pitch Attitude: [6.0, 3.0, 0.0, 20.0]
Roll Attitude: [6.0, 3.0, 0.0, 20.0]
Yaw Attitude: [6.0, 1.0, 0.35, 360.0]

"the angle PID runs on the fused IMU data to generate a desired rate of rotation. This rate of rotation feeds in to the rate PID which produces motor setpoints"
'''


def gen_pid_params(policy_cfg):
    mode = policy_cfg.mode
    parameters = []
    min_params = policy_cfg.pid.params.min_values
    max_params = policy_cfg.pid.params.max_values
    if mode == 'BASIC':
        # Pitch and Roll PD control
        num_control = 2
        for _ in range(num_control):
            P = np.random.uniform([min_params[0]], [max_params[0]])
            I = np.zeros(1)
            D = np.random.uniform([min_params[1]], [max_params[1]])
            parameters.append([P, I, D])

    elif mode == 'EULER':
        # Pitch Roll and Yaw PID Control
        num_control = 3
        for _ in range(num_control):
            P = np.random.uniform([min_params[0]], [max_params[0]])
            I = np.random.uniform([min_params[1]], [max_params[1]])
            D = np.random.uniform([min_params[2]], [max_params[2]])
            parameters.append([P, I, D])

    else:
        raise ValueError(f"Mode Not Supported {mode}")

    return parameters


class PidPolicy(Controller):
    def __init__(self, cfg):
        self.mode = cfg.mode
        self.pids = []

        # assert len(cfg.params.min_pwm) == len(cfg.params.equil)
        # assert len(cfg.params.max_pwm) == len(cfg.params.equil)

        self.min_pwm = cfg.params.min_pwm
        self.max_pwm = cfg.params.max_pwm
        self.equil = cfg.params.equil
        self.dt = cfg.params.dt
        self.numParameters = 0
        parameters = gen_pid_params(cfg)
        # order: pitch, roll, yaw, pitchrate, rollrate, yawRate or pitch roll yaw yawrate for hybrid or pitch roll yaw for euler
        if self.mode == 'BASIC':
            self.numpids = 2
            self.numParameters = 4
        elif self.mode == 'EULER':
            self.numpids = 3
            self.numParameters = 9
        else:
            raise ValueError(f"Mode Not Supported {self.mode}")

        for set in parameters:
            """
            def __init__(self, desired,
                 kp, ki, kd,
                 ilimit, dt, outlimit=np.inf,
                 samplingRate=0, cutoffFreq=-1,
                 enableDFilter=False):
             """
            P = set[0]
            I = set[1]
            D = set[2]
            self.pids += [PID(0, P, I, D, 1000, self.dt)]

    def get_action(self, state):

        # PIDs must always come in order of states then
        for i, pid in enumerate(self.pids):
            pid.update(state[i])

        def limit_thrust(pwm):  # Limits the thrust
            return np.clip(pwm, self.min_pwm, self.max_pwm)

        output = [0, 0, 0, 0]
        # PWM structure: 0:front right  1:front left  2:back left   3:back right
        '''Depending on which PID mode we are in, output the respective PWM values based on PID updates'''
        if self.mode == 'BASIC':
            output[0] = limit_thrust(self.equil[0] - self.pids[0].out + self.pids[1].out)
            output[1] = limit_thrust(self.equil[1] - self.pids[0].out - self.pids[1].out)
            output[2] = limit_thrust(self.equil[2] + self.pids[0].out - self.pids[1].out)
            output[3] = limit_thrust(self.equil[3] + self.pids[0].out + self.pids[1].out)
        elif self.mode == 'EULER':
            output[0] = limit_thrust(self.equil[0] - self.pids[0].out + self.pids[1].out + self.pids[2].out)
            output[1] = limit_thrust(self.equil[1] - self.pids[0].out - self.pids[1].out - self.pids[2].out)
            output[2] = limit_thrust(self.equil[2] + self.pids[0].out - self.pids[1].out + self.pids[2].out)
            output[3] = limit_thrust(self.equil[3] + self.pids[0].out + self.pids[1].out - self.pids[2].out)
        elif self.mode == 'HYBRID':
            output[0][0] = limit_thrust(self.equil[0] - self.pids[0].out + self.pids[1].out + self.pids[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.pids[0].out - self.pids[1].out - self.pids[5].out)
            output[0][2] = limit_thrust(self.equil[2] + self.pids[0].out - self.pids[1].out + self.pids[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.pids[0].out + self.pids[1].out - self.pids[5].out)
        elif self.mode == 'RATE':  # update this with the signs above
            output[0][0] = limit_thrust(self.equil[0] + self.pids[3].out - self.pids[4].out + self.pids[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.pids[3].out - self.pids[4].out - self.pids[5].out)
            output[0][2] = limit_thrust(self.equil[2] - self.pids[3].out + self.pids[4].out + self.pids[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.pids[3].out + self.pids[4].out - self.pids[5].out)
        elif self.mode == 'ALL':  # update this with the signs above
            output[0][0] = limit_thrust(
                self.equil[0] + self.pids[0].out - self.pids[1].out + self.pids[2].out + self.pids[3].out - self.pids[
                    4].out + self.pids[5].out)
            output[0][1] = limit_thrust(
                self.equil[1] - self.pids[0].out - self.pids[1].out - self.pids[2].out - self.pids[3].out - self.pids[
                    4].out - self.pids[5].out)
            output[0][2] = limit_thrust(
                self.equil[2] - self.pids[0].out + self.pids[1].out + self.pids[2].out - self.pids[3].out + self.pids[
                    4].out + self.pids[5].out)
            output[0][3] = limit_thrust(
                self.equil[3] + self.pids[0].out + self.pids[1].out - self.pids[2].out + self.pids[3].out + self.pids[
                    4].out - self.pids[5].out)
        return np.array(output)

    def reset(self):
        [p.reset() for p in self.pids]

    def update(self, states):
        '''

        :param states:
        :return:
        Order of states being passed: pitch, roll, yaw
        Updates the PID outputs based on the states being passed in (must be in the specified order above)
        Order of PIDs: pitch, roll, yaw, pitchRate, rollRate, yawRate
        '''
        assert len(states) == 3
        EulerOut = [0, 0, 0]
        for i in range(3):
            EulerOut[i] = self.pids[i].update(states[i])
        if self.mode == 'HYBRID':
            self.pids[3].update(EulerOut[2])
        if self.mode == 'RATE' or self.mode == 'ALL':
            for i in range(3):
                self.pids[i + 3].update(EulerOut[i])
