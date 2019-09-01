import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression

import numpy as np
import matplotlib.colors as color
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from dotmap import DotMap

import json
import datetime
import glob
import pandas
import os
import random as rand
import torch
import math
import sys

from PID import PID
from ExecuteTrain import getInput
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIDPolicy import policy

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

############################################################################
'''Dictionaries for indices of state inputs for simulation data vs ionocraft data'''
simDict = {'X': 0,'Y': 1,'Z': 2,'vx': 3,'vy': 4,'vz': 5,'yaw': 6,'pitch': 7,'roll': 8,
        'pitchRate': 9,'rollRate': 10, 'yawRate': 11}
ionoDict = {'pitchRate': 0,'rollRate': 1,'yawRate': 2,'pitch': 3,'roll': 4,'yaw': 5,
        'linax': 6,'linay': 7,'linaz': 8}
############################################################################
'''Class for bayesian optimization w.r.t. PID values and objective loss'''
class BOPID():
    def __init__(self, BOdict, PolicyDict):
        self.PID_Object = PID_Objective(BOdict, PolicyDict)
        self.PIDMODE = BOdict['POLICYMODE']
        evals = BOdict['ITERATIONS']
        zeros = [0,0,0]
        maximums = [300, 150, 20]
        if self.PIDMODE == 'EULER':
            self.n_parameters = 9
        elif self.PIDMODE == 'HYBRID':
            self.n_parameters = 12
        elif self.PIDMODE == 'RATE' or self.PIDMODE == 'ALL':
            self.n_parameters = 18
        else:
            print("Invalid PID mode selected")
            sys.exit(0)
        self.task = OptTask(f=self.PID_Object, n_parameters=self.n_parameters, n_objectives=1,
                    bounds=bounds(min=zeros * int(self.n_parameters / 3),max = maximums * int(self.n_parameters / 3)), task = {'minimize'}, vectorized=False)
                    #labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll', 'KP_yaw', 'KI_yaw', 'KD_yaw', 'KP_pitchRate', 'KI_pitchRate', 'KD_pitchRate', 'KP_rollRate',
                                    #'KI_rollRate', 'KD_rollRate', "KP_yawRate", "KI_yawRate", "KD_yawRate"])
        self.Stop = StopCriteria(maxEvals=evals)
        self.sim = BOdict['sim']

    def optimize(self):
        p = DotMap()
        p.verbosity = 1
        p.acq_func = EI(model = None, logs = None) #EI(model = None, logs = logs)
        p.model = regression.GP
        self.opt = opto.BO(parameters=p, task=self.task, stopCriteria=self.Stop)
        self.opt.optimize()
        print("Highest number of iterations: ", max(its))

    def getParameters(self, plotResults = False, printResults = False):
        log = self.opt.get_logs()
        losses = log.get_objectives()
        best = log.get_best_parameters()
        bestLoss = log.get_best_objectives()
        nEvals = log.get_n_evals()
        best = [matrix.tolist() for matrix in best] #can be a buggy line 

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

        if plotResults:
            plt.title("Evals vs Losses")
            plt.plot(list(range(nEvals)), losses[0])
            plt.show()
            scatter3d(Proll,Iroll,Droll, rollLoss, "Losses W.R.T. Roll PIDs")
            scatter3d(Ppitch,Ipitch,Dpitch, pitchLoss, "Losses W.R.T. Pitch PIDs")
            scatter3d(Pyaw,Iyaw,Dyaw, yawLoss, "Losses W.R.T. Yaw Rate PIDs")

        return best, bestLoss

####################################################################################
'''Main function for executing PID experiments in opto BO. General information'''
def PID_Objective(BOdict, PolicyDict):
    """
    Objective function of state data for a PID parameter tuning BO algorithm.
    Max flight time 10 seconds during rollouts. Operating at 25 Hz -> 250 Iterations.
    """

################################################################################
    '''Setting up Bayesian Optimization: model, initial conditions, parameters'''
    BOMODE = BOdict['BOMODE']
    BOOBJECTIVE = BOdict['BOOBJECTIVE']
    PIDMODE = BOdict['POLICYMODE']
    PolicyDict = PolicyDict
    model = BOdict['model']
    equil = BOdict['Equil']
    dt = BOdict['dt']
    sim = BOdict['sim']
    dataset = BOdict['data']

    sim = sim
    if sim:
        #If this is from the simulation, it will include positions in the three leftmost columns. Stack is still 3
        STATES = dataset[0]
        INPUTS = dataset[1]
        OUTPUTS = dataset[2]
    else:
        STATES, INPUTS = getInput(model)
    length = len(STATES)
    equil = equil
    min_pwm = 20000
    max_pwm = 65500
    dt = dt #data is at 25Hz
    pitchidx = simDict['pitch'] if sim else ionoDict['pitch']
    rollidx = simDict['roll'] if sim else ionoDict['roll']
    yawidx = simDict['yaw'] if sim else ionoDict['yaw']
    pitchRidx = simDict['pitchRate'] if sim else ionoDict['pitchRate']
    rollRidx = simDict['rollRate'] if sim else ionoDict['rollRate']
    yawRidx = simDict['yawRate'] if sim else ionoDict['yawRate']

###############################################################################
    '''General methods'''
    def get_initial_condition(): #gets an initial condition to kickstart BO rollouts
        validInput = False
        while (not validInput):
            randidx = np.random.randint(0, length)
            state = torch.from_numpy(STATES[randidx, :])
            action = torch.from_numpy(INPUTS[randidx, :])
            output = torch.from_numpy(OUTPUTS[randidx, :])
            if (abs(state[pitchidx]) < math.radians(10) and abs(state[rollidx]) < math.radians(10) and abs(state[yawidx]) < math.radians(15)):
                validInput = True
        return state, action, output

    def record_results(x, pLoss, rLoss, yLoss, itg, it): #saves PID parameters for piitch, roll, yaw and the losses associated with them for possible analysis later
        Ppitch.append(x[0,0])
        Ipitch.append(x[0,1])
        Dpitch.append(x[0,2])
        pitchLoss.append(pLoss * itg)
        Proll.append(x[0,3])
        Iroll.append(x[0,4])
        Droll.append(x[0,5])
        rollLoss.append((rLoss * itg))
        Pyaw.append(x[0,6]) #rate
        Iyaw.append(x[0,7])
        Dyaw.append(x[0,8])
        yawLoss.append((yLoss * itg))
        its.append(it)

    def devFromLstSqr(errors): #used in hurst exp. calculates deviations from least square line
        #Perform least square
        x = np.array(list(range(len(errors))))
        A = np.vstack([x, np.ones(len(errors))]).T
        m, c = np.linalg.lstsq(A, errors, rcond = None)[0]
        #Detrend the errors
        x = (m * x) + c
        resid = errors - x
        return resid

    def gaussianState(out):
        #Takes in a SINGLE prediction output from the probablistic (ensemble) network (means and variances) and returns a 9 element probablistic state
        assert np.shape(out)[1] == model.n_out #fix this to be more general
        assert np.shape(out)[0] == 1
        n_in = int(model.n_in_state / model.stack)
        mean = out[:, :n_in]
        var = out[:,n_in:]
        result = np.zeros(n_in)
        for i in range(n_in):
            result[i] = np.random.normal(mean[0, i], abs((var[0,i])) ** (1/2), 1)
        return result
###############################################################################
    '''Objective functions '''
    def IAE(x):
        SECONDS = 10 #Tuneable parameter. Number of seconds the rollout should simulate
        HERTZ = int(1/dt) #Not tuneable parameter. The frequency the data was operating on. Don't change unless we change the training frequency
        it = 5
        eWeight = 10
        tWeight = 2

        eRatio = eWeight / (eWeight + tWeight)
        tRatio = tWeight / (eWeight + tWeight)

        PolicyDict['PID'] = (x[0]).tolist()[0]
        PIDPolicy = policy(PolicyDict)

        iLoss = torch.tensor([0]).double()
        rLoss = 0
        pLoss = 0
        yLoss = 0
        time = 0

        maximumLoss = (SECONDS * HERTZ) * (math.radians(30)) * dt * 9 * 2
        itg = 1/maximumLoss #intuition: try best to "normalize" the outputs. Divide by the max amount of iLoss a rollout could've garnered
        max = math.radians(30)

        for j in range(it):
            state, action, outTarget = get_initial_condition()
            for i in range (SECONDS * HERTZ): #tuneable number of runs for each PID input. Remember: 4 of these are used for just getting a full input
                '''Pass into the model'''
                state = torch.reshape(state, (1, -1))
                action = torch.reshape(action, (1,-1))
                output = model.predict(state, action) #pre and post processing already handled
                if torch.isnan(torch.FloatTensor(output)).byte().any():
                    print("Ending this rollout due to nan value")
                    break
                if model.prob:
                    newState = torch.from_numpy(gaussianState(output)).double()
                else:
                    newState = output.double()

                if (abs(newState[pitchidx]) > max or abs(newState[rollidx]) > max or abs(newState[yawidx]) > max): #in case of system failure
                    #print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!"
                    #iLoss += 9 * max * ((SECONDS * HERTZ) - i) * dt#calculate by taking the max loss contributed by pitch, roll, yaw rate
                    break

                state = (state[0]).double()
                action = (action[0]).double()

                #still keeping track of the pitch, rolls, and yawRates to visually analyze stability
                pLoss += abs(newState[pitchidx].detach()) * dt
                rLoss += abs(newState[rollidx].detach()) * dt
                yLoss += abs(newState[yawidx].detach()) * dt

                #objective loss is calculated depending on the BO objective we are trying to achieve
                if BOOBJECTIVE == 'EULER':
                    iLoss += (abs(newState[pitchidx].detach()) + abs(newState[rollidx].detach()) + abs(newState[yawidx].detach())) #* dt
                elif BOOBJECTIVE == 'RATE':
                    iLoss += (abs(newState[pitchRidx].detach()) + abs(newState[rollRidx].detach()) + abs(newState[yawRidx].detach())) * dt
                elif BOOBJECTIVE == 'HYBRID':
                    iLoss += (abs(newState[pitchidx].detach()) + abs(newState[rollidx].detach()) + abs(newState[yawRidx].detach())) * dt
                elif BOOBJECTIVE == 'ALL':
                    iLoss += (abs(newState[pitchidx].detach()) + abs(newState[rollidx].detach()) + abs(newState[yawidx].detach()) + abs(newState[pitchRidx].detach()) + abs(newState[rollRidx].detach()) + abs(newState[yawRidx].detach())) * dt
                else:
                    print("Error: wrong objective mode selected.")
                    sys.exit(0)
                    break

                #update the PID outputs
                newStateList = newState.tolist()
                newStateList = [math.degrees(newStateList[pitchidx]), math.degrees(newStateList[rollidx]), math.degrees(newStateList[yawidx])]
                PIDPolicy.update(newStateList) #the policy updates only based on the pitch, roll, and yaw
                new_act = PIDPolicy.chooseAction().double()
                #state = newState
                #action = new_act

                #change the states and actions based on how much we have stacked
                nState = int(model.n_in_state/model.stack)
                nInput = int(model.n_in_input/model.stack)
                state = torch.cat((state.narrow(0,nState,(nState) * (model.stack - 1)), newState.detach()), 0)
                action = torch.cat((action.narrow(0,nInput,(nInput) * (model.stack -1)), new_act), 0)

            #add i+1 to time to track how much total time this policy took
            time += i+1
        #iLoss = iLoss * itg
        iLoss = (iLoss / (i + 1))/it #iloss per frame per iteration
        #print("iLoss:", iLoss, " after ", i + 1, " iterations")
        #print("")

        record_results(x, pLoss, rLoss, yLoss, itg, i+1)
        #total loss is divided by 3*max to normalize it. Second term penalizes rollouts with less flight time
        totLoss = ((iLoss/(3 * max))) #- ((tRatio*time)/ (SECONDS * HERTZ * it))
        return totLoss

    def Hurst(x):
        '''OUT OF DATE! MUST UPDATE TO SUPPORT NEW IMPLEMENTATIONS

        The Hurst exponent is an attribute that, when computed, represents
        the presence of long-term correlation among error signals. Generally an exponent
        value of:
            - alpha = .5  -> white noise signal
            - alpha > .5 -> correlation in time series signal
            - alpha < .5 -> anti-correlation in time series signal
        As a result, alpha of .5 indicates good PID fitness

        Steps to calculate Hurst exponent:
         1. Find E(k) (the sum of mean centered signals up to k)
         2. Divide E(k) into boxes of equal length n (min 10 max N/4)
         3. For each box, perform linear least squares
         4. Subtract E(k) by the local trend
         5. Calculate F(n) (sqrt mean squared deviation from local trend of all points of all boxes)
         6. Repeat over many different size boxes n
         7. Find log-log plot of F(n) versus n.
         8. Calculate gradient of the line. This is the value of the Hurst exponent'''

        pitchPID = PID(0, x[0,0], x[0,1], x[0,2], 20, dt) #why are we limiting the i term????
        rollPID = PID(0, x[0,3], x[0,4], x[0,5], 20, dt)
        yawPID = PID()
        yawratePID = PID(0, x[0,6], x[0,7], x[0,8], 360, dt)

        rErrors = []
        pErrors = []
        yErrors = []
        boxes = [10, 12, 20, 30, 40, 60]
        state, action = get_initial_condition()

        for i in range(240):
            if state[3] >= 30 or state[4] >= 30: #in case of system failure
                print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
                iLoss += 60 * (250 - (i+ 1))
                break

            '''Pass into the model'''
            output = model.predict(state, action) #pre and post processing already handled
            assert not torch.isnan(torch.FloatTensor(output)).byte().any()
            newState = torch.from_numpy(output)

            if rErrors == []:
                pErrors = np.array([abs(newState[3].detach())])
                rErrors = np.array([abs(newState[4].detach())])
                yErrors = np.array([abs(newState[2].detach())])
            else:
                pErrors = pErrors.hstack(np.array([pErrors[i - 1] + abs(newState[3].detach())]))
                rErrors = rErrors.hstack(np.array([rErrors[i - 1] + abs(newState[4].detach())]))
                yErrors = yErrors.hstack(np.array([yErrors[i - 1] + abs(newState[2].detach())]))

            pitchPID.update(newState[3])
            rollPID.update(newState[4])
            yawratePID.update(newState[2])
            new_act = new_action(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil)
            state = torch.cat((state.narrow(0,9,18), newState.detach()), 0)
            action = torch.cat((action.narrow(0,4,8), new_act), 0)

        pErrors = pErrors - np.mean(pErrors)
        rErrors = rErrors - np.mean(rErrors)
        yErrors = yErrors - np.mean(yErrors)
        n = []
        pF = []
        rF = []
        yF = []
        N = len(pErrors)

        for size in boxes:
            i = 0
            pbox = []
            rbox = []
            ybox = []
            while (i < len(pErrors)):
                pboxResults = devFromLstSqr(pErrors[i:i + size])
                rboxResults = devFromLstSqr(rErrors[i: i + size])
                yboxResults = devFromLstSqr(yErrors[i: i + size])
                if pbox == []:
                    pbox = pboxResults
                    rbox = rboxResults
                    ybox = yboxResults
                else:
                    pbox = np.hstack(pbox, pboxResults)
                    rbox = np.hstack(robx, rboxResults)
                    ybox = np.hstack(ybox, yboxResults)
                i += size
            pFvalue =  (np.sum(np.square(pbox)) / N) ** (1/2)
            rFvalue = (np.sum(np.square(rbox)) / N) ** (1/2)
            yFvalue = (np.sum(np.square(ybox)) / N) ** (1/2)

            n += [size]
            pF += [pFvalue]
            rF += [rFvalue]
            yF += [yFvalue]

        logn = np.log(np.array(n))
        logpF = np.log(np.array(pF))
        logrF = np.log(np.array(rF))
        logyF = np.log(np.array(yF))

        A = np.vstack([logn, np.ones(len(errors))]).T
        alphap, c = np.linalg.lstsq(A, logpF, rcond = None)[0]
        alphar, c = np.linalg.lstsq(A, logrF, rcond = None)[0]
        alphay, c = np.linalg.lstsq(A, logyF, rcond = None)[0]

        return ((alphap - .5)**2 + (alphar - .5) ** 2 + (alphay - .5)**2) ** (1/2)

###############################################################################
    '''Wrapper for objective functions'''

    def objective(x):
        """
        Assess the objective value of a trajectory X.
        """

        # various modes of the objective function.
        if BOMODE == 'IAE':
            return IAE(x)
        elif BOMODE == 'HURST':
            return Hurst(x)
        else:
            print("Invalid BO mode selected")
            sys.exit(0)

    return objective

################################################################################
'''Recording results for plotting and saving'''
Proll = []
Iroll = []
Droll = []
rollLoss = []
Ppitch = []
Ipitch = []
Dpitch = []
pitchLoss = []
Pyaw = [] #rate
Iyaw = []
Dyaw = []
yawLoss = []
its = []
################################################################################
'''Plotting and visualizing results. Makes a 3D grid plotting individual pitch, roll, yaw losses wrt PID parameters'''

def scatter3d(x,y,z, cs, name, colorsMap='viridis'):
    cm = plt.get_cmap(colorsMap)
    cNorm = color.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    plt.title(name)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    ax.set_xlabel('Proportional')
    ax.set_ylabel('Integral')
    ax.set_zlabel('Derivative')
    ax.set_title(name)
    plt.show()
