import os
import sys
from dotmap import DotMap

import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression

import numpy as np
import torch
import math

from learn.control.pid import PID
import logging
import hydra

log = logging.getLogger(__name__)

# Bayesian Optimization Class build on opto

'''Defines a class and numerous methods for performing Bayesian Optimization
    Variables: - PID_Object: object to Opto object to optimize
               - PIDMODE: PID policy mode (euler...rate etc)
               - n_parameters: number of total PID parameters for the policy we are testing
               - task: an opttask that is to be optimized through the Opto library
               - Stop: stop criteria for BO
               - sim: boolean signalling whether we are using a simulation (includes position in state)
    Methods: - optimize: uses opto library and EI acquisition to perform BO
             - getParameters: returns the results of bayesian optimization
'''


def temp_objective(x):
    return np.random.rand(1)


class BOPID():
    def __init__(self, bo_cfg, policy_cfg):
        # self.Objective = SimulationOptimizer(bo_cfg, policy_cfg)
        self.PIDMODE = policy_cfg.mode
        evals = bo_cfg.iterations
        zeros = [0, 0, 0]
        maximums = [300, 150, 20]
        if self.PIDMODE == 'BASIC':
            self.n_parameters = 4
        if self.PIDMODE == 'EULER':
            self.n_parameters = 9
        elif self.PIDMODE == 'HYBRID':
            self.n_parameters = 12
        elif self.PIDMODE == 'RATE' or self.PIDMODE == 'ALL':
            self.n_parameters = 18
        else:
            print("Invalid PID mode selected")
            sys.exit(0)
        self.task = OptTask(f=temp_objective, n_parameters=self.n_parameters, n_objectives=1,
                            bounds=bounds(min=zeros * int(self.n_parameters / 3),
                                          max=maximums * int(self.n_parameters / 3)), task={'minimize'},
                            vectorized=False)
        # labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll', 'KP_yaw', 'KI_yaw', 'KD_yaw', 'KP_pitchRate', 'KI_pitchRate', 'KD_pitchRate', 'KP_rollRate',
        # 'KI_rollRate', 'KD_rollRate', "KP_yawRate", "KI_yawRate", "KD_yawRate"])
        self.Stop = StopCriteria(maxEvals=evals)
        self.sim = bo_cfg.sim

    def optimize(self):
        p = DotMap()
        p.verbosity = 1
        p.acq_func = EI(model=None, logs=None)
        p.model = regression.GP
        self.opt = opto.BO(parameters=p, task=self.task, stopCriteria=self.Stop)
        self.opt.optimize()


    def getParameters(self, plotResults=False, printResults=False):
        log = self.opt.get_logs()
        losses = log.get_objectives()
        best = log.get_best_parameters()
        bestLoss = log.get_best_objectives()
        nEvals = log.get_n_evals()
        best = [matrix.tolist() for matrix in best]  # can be a buggy line

        if printResults:
            print("Best PID parameters found with loss of: ", np.amin(bestLoss), " in ", nEvals, " evaluations.")
            print("Pitch:   Prop: ", best[0], " Int: ", best[1], " Deriv: ", best[2])
            print("Roll:    Prop: ", best[3], " Int: ", best[4], " Deriv: ", best[5])
            print("Yaw:     Prop: ", best[6], " Int: ", best[7], " Deriv: ", best[8])
            if self.PIDMODE == 'HYBRID':
                print("YawRate: Prop: ", best[9], " Int: ", best[10], "Deriv: ", best[11])
            if self.PIDMODE == 'RATE' or self.PIDMODE == 'ALL':
                print("PitchRt: Prop: ", best[9], " Int: ", best[10], " Deriv: ", best[11])
                print("RollRate:Prop: ", best[12], " Int: ", best[13], " Deriv: ", best[14])
                print("YawRate: Prop: ", best[15], " Int: ", best[16], "Deriv: ", best[17])

        return best, bestLoss

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


class policy():
    def __init__(self, cfg):
        self.mode = cfg.mode
        self.PID = []
        self.numPIDs = 0
        assert len(cfg.min_pwm) == len(cfg.equil)
        assert len(cfg.max_pwm) == len(cfg.equil)

        self.min_pwm = cfg.min_pwm
        self.max_pwm = cfg.max_pwm
        self.equil = cfg.equil
        self.dt = cfg.dt
        self.numParameters = 0
        parameters = policyDict['PID']
        # order: pitch, roll, yaw, pitchrate, rollrate, yawRate or pitch roll yaw yawrate for hybrid or pitch roll yaw for euler
        if self.mode == 'EULER':
            self.numParameters = 9
        elif self.mode == 'HYBRID':
            self.numParameters = 12
        elif self.mode == 'RATE' or self.mode == 'ALL':
            self.numParameters = 18
        assert len(parameters) == self.numParameters
        self.numPIDs = int(self.numParameters / 3)

        for i in [3 * i for i in list(range(self.numPIDs))]:
            self.PID += [PID(0, parameters[i], parameters[i + 1], parameters[i + 2], 1000, self.dt)]

    def chooseAction(self):
        def limit_thrust(PWM):  # Limits the thrust
            return np.clip(PWM, self.min_pwm, self.max_pwm)

        output = [0, 0, 0, 0]
        # PWM structure: 0:front right  1:front left  2:back left   3:back right
        '''Depending on which PID mode we are in, output the respective PWM values based on PID updates'''
        if self.mode == 'EULER':
            output[0] = limit_thrust(self.equil[0] - self.PID[0].out + self.PID[1].out + self.PID[2].out)
            output[1] = limit_thrust(self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[2].out)
            output[2] = limit_thrust(self.equil[2] + self.PID[0].out - self.PID[1].out + self.PID[2].out)
            output[3] = limit_thrust(self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[2].out)
        elif self.mode == 'HYBRID':
            output[0][0] = limit_thrust(self.equil[0] - self.PID[0].out + self.PID[1].out + self.PID[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[5].out)
            output[0][2] = limit_thrust(self.equil[2] + self.PID[0].out - self.PID[1].out + self.PID[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[5].out)
        elif self.mode == 'RATE':  # update this with the signs above
            output[0][0] = limit_thrust(self.equil[0] + self.PID[3].out - self.PID[4].out + self.PID[5].out)
            output[0][1] = limit_thrust(self.equil[1] - self.PID[3].out - self.PID[4].out - self.PID[5].out)
            output[0][2] = limit_thrust(self.equil[2] - self.PID[3].out + self.PID[4].out + self.PID[5].out)
            output[0][3] = limit_thrust(self.equil[3] + self.PID[3].out + self.PID[4].out - self.PID[5].out)
        elif self.mode == 'ALL':  # update this with the signs above
            output[0][0] = limit_thrust(
                self.equil[0] + self.PID[0].out - self.PID[1].out + self.PID[2].out + self.PID[3].out - self.PID[
                    4].out + self.PID[5].out)
            output[0][1] = limit_thrust(
                self.equil[1] - self.PID[0].out - self.PID[1].out - self.PID[2].out - self.PID[3].out - self.PID[
                    4].out - self.PID[5].out)
            output[0][2] = limit_thrust(
                self.equil[2] - self.PID[0].out + self.PID[1].out + self.PID[2].out - self.PID[3].out + self.PID[
                    4].out + self.PID[5].out)
            output[0][3] = limit_thrust(
                self.equil[3] + self.PID[0].out + self.PID[1].out - self.PID[2].out + self.PID[3].out + self.PID[
                    4].out - self.PID[5].out)
        return torch.FloatTensor(output)

    def update(self, states):
        '''Order of states being passed: pitch, roll, yaw'''
        '''Updates the PID outputs based on the states being passed in (must be in the specified order above)'''
        '''Order of PIDs: pitch, roll, yaw, pitchRate, rollRate, yawRate'''
        assert len(states) == 3
        EulerOut = [0, 0, 0]
        for i in range(3):
            EulerOut[i] = self.PID[i].update(states[i])
        if self.mode == 'HYBRID':
            self.PID[3].update(EulerOut[2])
        if self.mode == 'RATE' or self.mode == 'ALL':
            for i in range(3):
                self.PID[i + 3].update(EulerOut[i])


######################################################################
@hydra.main(config_path='conf/simulate.yaml')
def optimizer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    sim = BOPID(cfg.bo, cfg.policy)
    sim.optimize()


if __name__ == '__main__':
    sys.exit(optimizer())
