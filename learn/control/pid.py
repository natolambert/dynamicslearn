
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
                    ilimit, dt, outlimit = np.inf,
                    samplingRate = 0, cutoffFreq = -1,
                    enableDFilter = False):

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
        self.samplingRate = samplingRate    # sample rate is for filtering

        self.cutoffFreq = cutoffFreq
        self.enableDFilter = enableDFilter

        if cutoffFreq != -1 or enableDFilter:
            raise NotImplementedError('Have not implemnted filtering yet')

    def update(self, measured):

        # init
        self.out = 0.

        # update error
        self.error_prev = self.error

        # calc new error
        self.error = self.desired - measured

        # proportional gain is easy
        self.out += self.kp*self.error

        # calculate deriv term
        self.deriv = (self.error-self.error_prev) / self.dt

        # filtter if needed (DT function_)
        if self.enableDFilter:
            print('Do Filter')
            self.deriv = self.deriv

        # calcualte error value added
        self.out += self.deriv*self.kd

        # accumualte normalized eerror
        self.integral = self.error*self.dt

        # limitt the integral term
        if self.ilimit !=0:
            self.integral = np.clip(self.integral,-self.ilimit, self.ilimit)

        self.out += self.ki*self.integral

        # limitt the total output
        if self.outlimit !=0:
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
# class to mimic the PID structure onboard the crazyflie
class crazyPID():
    """
    Class for bootstrapping PID controllers off of a learned dynamics model.
    """
    def __init__(self, equil, dt, minpwm = 0, maxpwm = 65535, out_lim = 5000,
                att_pitch = [], att_roll = [], att_yaw = [],
                rate_pitch = [], rate_roll = [], rate_yaw = []):

        self.equil = equil
        self.dt = dt
        self.minpwm = 0
        self.maxpwm = 65535

        self.output = equil

        # PIDs
        self.PID_att_pitch = []
        self.PID_att_roll = []
        self.PID_att_yaw = []
        self.PID_rate_pitch = []
        self.PID_rate_roll = []
        self.PID_rate_yaw = []

        # Above, all of the last six inputs being att_pitch etc are lists of length 5
        # Axis Mode: [KP, KI, KD, iLimit]
        if att_pitch != []:
            self.PID_att_pitch = PID(0, att_pitch[0],
                                        att_pitch[1],
                                        att_pitch[2],
                                        att_pitch[3], dt)

        if att_roll != []:
            self.PID_att_roll = PID(0, att_roll[0],
                                        att_roll[1],
                                    att_roll[2],
                                        att_roll[3], dt)

        if att_yaw != []:
            self.PID_att_yaw = PID(0, att_yaw[0],
                                        att_yaw[1],
                                        att_yaw[2],
                                        att_yaw[3], dt)

        if rate_pitch != []:
            self.PID_rate_pitch = PID(0, rate_pitch[0],
                                        rate_pitch[1],
                                        rate_pitch[2],
                                        rate_pitch[3], dt)

        if rate_roll != []:
            self.PID_rate_roll = PID(0, rate_roll[0],
                                        rate_roll[1],
                                        rate_roll[2],
                                        rate_roll[3], dt)

        if rate_yaw != []:
            self.PID_rate_yaw = PID(0, rate_yaw[0],
                                        rate_yaw[1],
                                        rate_yaw[2],
                                        rate_yaw[3], dt)

        # create list of 'active' PIDs
        self.PIDs = []
        if self.PID_att_pitch != []: self.PIDs.append(self.PID_att_pitch)
        if self.PID_att_roll != []: self.PIDs.append(self.PID_att_roll)
        if self.PID_att_yaw != []: self.PIDs.append(self.PID_att_yaw)
        if self.PID_rate_pitch != []: self.PIDs.append(self.PID_rate_pitch)
        if self.PID_rate_roll != []: self.PIDs.append(self.PID_rate_roll)
        if self.PID_rate_yaw != []: self.PIDs.append(self.PID_rate_yaw)

        if len(self.PIDs) == 3:
            print("INIT PID IN ATTITUDE MODE")
            self.mode = 1
        elif len(self.PIDs) == 6:
            print("INIT PID IN ATTITUDE+RATE MODE")
            self.mode = 0

        def update(self, x):
            """
            This function will take in the current state, and update the PID's output.

            Takes x: 9 dimensional state
            Returns u: 4 dimensional action
            """

            def limit_thrust(PWM):
                """
                Limits thrust, can be adjusted for different robots
                """
                return np.clip(PWM, self.minpwm, self.maxpwm)

            # Update Attitude PIDs first
            out_pitch = self.PID_att_pitch.update(x[3])
            out_roll = self.PID_att_roll.update(x[4])
            out_yaw = self.PID_att_yaw.update(x[5])

            # Pass their outputs into the rate PIDs, and update
            out_pitch_rate = self.PID_rate_pitch.update(out_pitch)
            out_roll_rate = self.PID_rate_roll.update(out_roll)
            out_yaw_rate = self.PID_rate_yaw.update(out_yaw)

            # Update output from attitude PIDS
            if self.mode == 1:
                self.output[0] = limit_thrust(
                    self.equil[0] + self.PID_att_pitch.out + self.PID_att_yaw.out)
                self.output[1] = limit_thrust(
                    self.equil[1] - self.PID_att_roll.out - self.PID_att_yaw.out)
                self.output[2] = limit_thrust(
                    self.equil[2] - self.PID_att_pitch.out + self.PID_att_yaw.out)
                self.output[3] = limit_thrust(
                    self.equil[3] + self.PID_att_roll.out - self.PID_att_yaw.out)

            # Update output from Rate PIDs, which were updated from Attitude setpoints
            elif self.mode == 0:
                self.output[0] = limit_thrust(
                    self.equil[0] + self.PID_rate_pitch.out + self.PID_rate_yaw.out)
                self.output[1] = limit_thrust(
                    self.equil[1] - self.PID_rate_roll.out - self.PID_rate_yaw.out)
                self.output[2] = limit_thrust(
                    self.equil[2] - self.PID_rate_pitch.out + self.PID_rate_yaw.out)
                self.output[3] = limit_thrust(
                    self.equil[3] + self.PID_rate_roll.out - self.PID_rate_yaw.out)

            return self.output

