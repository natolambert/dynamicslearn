'''Uses all of the classes and functions we have defined thus far to make a wrapper function.

Steps:
    - Gathers data from quadcopter simulation from initial state inputs from execute train
    - Uses data from sim to train ensemble neural network with optimal parameters found previously
    - Uses the ensemble neural network in bayesian optimization to find a new set of optimal PID parameters
    - Uses those PID parameters to form a policy to pass into the simulation from initial state
    - Repeat '''
##############################################################################
'''Imports'''
import numpy as np
import torch
import torch.nn as nn
import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from Initial import radToDeg
import sys
import math

from Initial import getOptimalNNTrainParams, getState
from PIDPolicy import policy
from CrazyFlieSim import CrazyFlie
from EnsembleNN import EnsembleNN
from PID_BO_iono import BOPID, scatter3d
from RolloutAnalysis import Analysis
from operator import add
from operator import sub

##############################################################################
'''Tuneable parameters'''

dt = 1/75 #75 Hertz frequency
EQUILIBRIUM = [20000, 20000, 20000, 20000]
POLICYMODE = 'EULER'
CYCLES = 25

policyDict = { 'PID': None,
               'PolicyMode':POLICYMODE,
               'dt': dt,
               'Equil': EQUILIBRIUM,
               'min_pwm': 20000,
               'max_pwm': 65500}

simulatorDict = {'policy': None,
                 'initStates': None,
                 'simFlights': 10,
                 'visuals': False,
                 'maxRollout': 1500,
                 'addNoise': True,
                 'maxAngle': 30}

analysisDict = {'stack': 3,
                'data': None,
                'dX': 12,
                'dU': 4,
                'dT': 12,
                'dt': dt,
                'rewardTime': True,
                'rewardSSE': True,
                'rewardSettle': True,
                'rewardOver': True,
                'SETTLESECONDS': 5, #how many seconds euler angle needs to remain constant to be considered settled
                'SETTLEBENCHMARK' : 5, # maximum number of seconds for euler angle to be considered a "good" settling time
                'MAXANGLE': .033 * math.radians(30), #max deviation from 0 to considered a settled state
                'MINIMUMFRAMES': 1300} #minimum frames to consider a flight a good flight

BOdict = {'model': None,
          'BOMODE': 'IAE',
          'BOOBJECTIVE': 'EULER',
          'POLICYMODE': 'EULER',
          'ITERATIONS': 1,
          'Equil': EQUILIBRIUM,
          'dt': dt,
          'sim': True,
          'data': None,
          'path': None}

'''Model Training Parameters'''
EPOCHS = 5#10#number of epochs training the network in each cycle #5 if noisevector 10 if not
LR = .0001 #NOT CURRENTLY BEING USED
STEP = 12 #number of epochs to train before decaying the LR #use 6 if adding noisevector in simulation...30 if not
DECAY = .65 #amount to decay the learning rate by
KCLUSTER = True #toggles whether or not to use Kclustering on the data gathered from simulations
NUMCLUSTERS = 6 #if kcluster is toggled True, specifies how many clusters to cluster the dataset into
PADDING = 50 #add this many sampling points to each of the clusters to ideally get closer to the ideal split ratio
DECAYLR = True #toggles whether to decay the learning rate over time
MAXDATASIZE = 2500
INCLUDEPOSITION = True
NSTATE = 9
NINPUT = 4
NSTACK = 3
path = "EnsembleBOModelCycle3.txt"

################# INITIAL CONDITIONS/DECLARATIONS ###############
if POLICYMODE == 'EULER':
    policyDict['PID'] = [np.random.randint(0, 100) for i in range(9)]
elif POLICYMODE == 'HYBRID':
    policyDict['PID'] = [np.random.randint(0, 100) for i in range(12)]
elif POLICYMODE == 'RATE' or POLICYMODE == 'ALL':
    policyDict['PID'] = [np.random.randint(0, 100) for i in range(18)]
else:
    print("Error. Invalid policy mode selected.")
    sys.exit(0)

nn_params, train_params = getOptimalNNTrainParams(INCLUDEPOSITION, NSTATE, NINPUT, NSTACK, epochs = EPOCHS)
ensemble = EnsembleNN(nn_params)
ensemble.init_weights_orth()

'''Logging data'''
trainingLoss = [] #model
testingLoss = []
objectiveLoss = [] #bayesian optimization
simulationTime = [] #simulation flight time over cycles
simstd = [] #simulation flight time variance
simavg = [] #average error of rollouts
offsets = [] #used to keep track of individual flights in overall dataset
PIDParameters = [[] for p in range(len(policyDict['PID']))] #to keep track of the PID parameters explored
allData = None #stores all of the data taken from simulation flights. Initialized as None
firstCycle = True #indicates whether this is the first cycle
stepCounter = 0 #tracks number of epochs we have trained the model

############ HELPER TO GRAPH 2D PROJECTIONS #######
'''Helper to graph 2D projections onto 3D plane. Used to see how log distribution of pitch, roll, yaw changes over cycles'''
def twoOnThreeGraph(dataset, offsets, sizes, title):
    maxSize = max(sizes)
    resolution = 60
    dataset = np.degrees(dataset)
    x = np.linspace(-30, 30, resolution) #each "bin has a width of (30 + 30) / 60 = 1"
    y = []
    z = []
    #make a loop that basically makes a list of np arrays...each one corresponding to a different point in the cycle
    for i in range(len(offsets)):
        newy = np.ones(resolution) * (i + 1)
        newz = np.zeros(resolution)
        insert = dataset[offsets[i]:offsets[i] + sizes[i]]
        for j in list(range(insert.size)): #sort each data point into one of the bins by rounding it to the nearest integer.
            newz[30 - int(round(insert[j]))] += 1 #THIS ONLY WORKS BECAUSE WE ARE SORTING INTO INTEGER 1 SIZE BINS
        y += [newy]
        z += [newz]
    #map every z value to 0 if 0 and the log of the value if log....add a super small value so that values equal to 1 don't get mapped to zero either
    def logMap(x):
        if x == 0:
            return 0
        else:
            return math.log(x, 10)

    z = [[logMap(elem) for elem in lst] for lst in z]

    pl.figure()
    ax = pl.subplot(projection='3d')
    for i in range(len(y)):
        ax.plot(x, y[i], z[i], color = 'r')

    ax.set_title(title)
    ax.set_xlabel('Euler Angle Value')
    ax.set_zlabel('LogBase10-Scale Number of Occurrences')
    ax.set_ylabel('Cycle Number')
    plt.show()

################# MAIN LOOP #####################

for i in range(CYCLES):
    print("")
    print("")
    print("")
    print("####################### STARTING CYCLE NUMBER ", i + 1, "###################################")
    print("")
    print("Gathering data from simulation...")

    ############### SIMULATION ###################
    pol = policy(policyDict) #initializes a policy
    simulator = CrazyFlie(dt) #initializes a simulator
    simulatorDict['policy'] = pol
    simulatorDict['initStates'] = getState() #initial condition for simulation
    simTimeTotal, avgTime, simVar, avgError = simulator.test_policy(simulatorDict)
    print("")
    print("Gathered data from ", simulatorDict['simFlights'], " rollouts with total flight time of: ", simTimeTotal, " average time ticks: ", avgTime, " average error per frame: ", avgError)
    newData = simulator.get_recent_data() #structure: s0(12), a(4), s1(12), r(1) #new data retrieved from simulation
    assert not np.any(np.isnan(newData)) #check for nans in dataset

    ############# ANALYSIS ####################
    analysisDict['data'] = newData
    dataAnalysis = Analysis(analysisDict)
    dataset = dataAnalysis.stackOriginal()
    allData = np.hstack(dataset) if firstCycle else np.vstack((allData, np.hstack(dataset))) #appends newdata to all data for later analysis
    firstCycle = False
    extra = dataAnalysis.extraData() #performs analysis on settle times, overshoot etc
    if len(extra) == 3: #if we returned extra data
        dataset = dataAnalysis.stackThree(dataset, extra)

    ########## MODEL TRAINING ###################
    print("")
    print("Training model...")
    print("")
    if DECAYLR: #condition to check if we should change the LR. We don't know whether the mod will equal 0...train_cust trains it for multiple epochs
        if (stepCounter >= STEP) and stepCounter != 0:
            for params in train_params:
                params['lr'] = params['lr'] * DECAY
            stepCounter = stepCounter - STEP #modulus for some reason did not work
            STEP *= 2
            print("LR UPDATE: Network 1 learning rate has been changed to: ", train_params[0]['lr'])
            print("")
    size = min(MAXDATASIZE, dataset[0].shape[0]) #determines how big of a dataset to use for training/testing
    dataidx = np.random.choice(dataset[0].shape[0], size) #randomly sample
    dataset= (dataset[0][dataidx, :], dataset[1][dataidx,:], dataset[2][dataidx,:])
    testError, trainError = ensemble.train_cust(dataset, train_params, numClusters = NUMCLUSTERS, cluster = KCLUSTER, datasize = MAXDATASIZE, padding = PADDING) #the value returned is the average testing/training loss over ALL epochs across ALL networks in ensemble
    for netnum, network in enumerate(ensemble.networks):
        if any(torch.isnan(val).byte().any() for val in network.state_dict().values()): #check for Nans in any of the weights of the networks
            print("Error: Nan value in state dict of network number: ", netnum + 1, " (not zero indexed).")
            sys.exit(0)

    ########## BAYESIAN OPTIMIZATION #############
    print("")
    print("Running BO on PID Parameters and Model...")
    print("")
    BOdict['model'] = ensemble #use the model we just trained
    BOdict['data'] = dataset #pass in the new dataset we just got for initial conditions...consider passing in allData instead?
    BO = BOPID(BOdict, policyDict)
    BO.optimize()
    newParams, BOloss = BO.getParameters(False, False) #get the new optimal parameters...pass in falses to not plot nor print results
    print("New PID parameters found after ", BOdict['ITERATIONS'], " iterations with objective loss of: ", BOloss)
    policyDict['PID'] = newParams
    stepCounter += EPOCHS

    ######### STORING ANALYSIS VALUES ###############3#
    for pidIndex in range(len(policyDict['PID'])):
        PIDParameters[pidIndex] += [policyDict['PID'][pidIndex]] #store optimal PID parameters
    simulationTime += [simTimeTotal]
    simavg += [[avgTime, avgError]]
    simstd += [simVar**(1/2)]
    testingLoss += [testError]
    trainingLoss += [trainError]
    objectiveLoss += [BOloss]
    if offsets == []:
        offsets += [0]
    else:
        offsets += [offsets[i-1] + simulationTime[i-1]]

ensemble.save_model(path)


###################### GRAPHS AND DISPLAY RESULTS ##############################

simavgarray = np.array(simavg)
avgError = (simavgarray[:, 1].T).tolist()
avgTime = (simavgarray[:,0].T).tolist()
print("")
#plotting the average flight time wrt to cycles
# minus 1 offset because i's generation of PID parameters corresponds to i+1's simulation results
plt.plot(list(range(CYCLES-1)), avgTime[1:], 'r--')
plt.plot(list(range(CYCLES-1)), list(map(add, avgTime[1:], simstd[1:])), 'b--')
plt.plot(list(range(CYCLES-1)), list(map(sub, avgTime[1:], simstd[1:])), 'b--')
plt.show()

#plotting average error of simulation wrt cycles
plt.plot(list(range(CYCLES-1)), avgError[1:], 'g--')
plt.title("Simulation error (y) over cycles (x)")
plt.show()

#Plotting relationship between BO loss and respective simulation time
plt.plot(objectiveLoss[:-1], avgError[1:], 'ro')
plt.title("Average simulation error (y) with respect to BO objective loss (x)")

#Distributions of euler angles over time
twoOnThreeGraph(allData[:, 7].T, offsets, simulationTime, 'Pitch Distributions over Cycles (Not Including Failure Points)')
twoOnThreeGraph(allData[:, 8].T, offsets, simulationTime, 'Roll Distributions over Cycles (Not Including Failure Points)')
twoOnThreeGraph(allData[:, 6].T, offsets, simulationTime, 'Yaw Distributions over Cycles (Not including Failure Poionts)')

#Training and testing loss
plt.plot(list(range(CYCLES)), trainingLoss, 'r--', list(range(CYCLES)), testingLoss, 'b--')
plt.title("Training and Testing Loss of Model(y) over Cycles(x)")
red = mpatches.Patch(color = 'red', label = 'Training')
blue = mpatches.Patch(color = 'blue', label = 'Testing')
plt.legend(handles = [red, blue])
plt.show()

#3D scatter plots of PID parameters w.r.t. simulation time
scatter3d(PIDParameters[0], PIDParameters[1], PIDParameters[2], avgError, "Pitch Parameters w.r.t. SIMULATION ERROR")
scatter3d(PIDParameters[3], PIDParameters[4], PIDParameters[5], avgError, "Roll Parameters w.r.t. SIMULATION ERROR")
scatter3d(PIDParameters[6], PIDParameters[7], PIDParameters[8], avgError, "Yaw Parameters w.r.t. SIMULATION ERROR")
if POLICYMODE == 'HYBRID':
    scatter3d(PIDParameters[9], PIDParameters[10], PIDParameters[11], avgError, "Yaw Rate Parameters w.r.t. SIMULATION ERROR")
if POLICYMODE == 'RATE':
    scatter3d(PIDParameters[9], PIDParameters[10], PIDParameters[11], avgError, "Pitch Rate Parameters w.r.t. SIMULATION ERROR")
    scatter3d(PIDParameters[12], PIDParameters[13], PIDParameters[14], avgError, "Roll Rate Parameters w.r.t. SIMULATION ERROR")
    scatter3d(PIDParameters[15], PIDParameters[16], PIDParameters[17], avgError, "Yaw Rate Parameters w.r.t. SIMULATION ERROR")

######################## PRINT SUMMARY TO SCREEN #############################

print("########################     Summary:    #################################")
print("")
print("Total Cycles: ", CYCLES, "   BOIterations: ", BOdict['ITERATIONS'], "    Epochs: ", EPOCHS,"      SimFlights: ", simulatorDict['simFlights'])
print("Policy mode: ", policyDict['PolicyMode'], " BO Objective: ", BOdict['BOOBJECTIVE'])
highestFrames = max(map(lambda x: x[0], simavg))
lowestLoss = min([lst[1] for lst in simavg if lst[0] == highestFrames])
c = simavg.index([highestFrames, lowestLoss])
print("Max average simulation flight time: ", highestFrames, " found at cycle number: ", c + 1, " with loss of ", lowestLoss)
print("Minimum model test loss: ", min(testingLoss), " found at cycle number: ", testingLoss.index(min(testingLoss)) + 1)
print("Minimum model train loss: ", min(trainingLoss), "found at cycle number: ", trainingLoss.index(min(trainingLoss)) + 1)
print("Minimum BO objective loss: ", min(objectiveLoss), "found at cycle number ", objectiveLoss.index(min(objectiveLoss)) + 1)
print("")
slope, intercept, r_value, p_value, std_error = stats.linregress(objectiveLoss[:-1], avgError[1:])
print("Linear Regression on BO objective loss vs Flight Simulation Error had slope: ", slope, " intercept: ", intercept, " and r value of ", r_value)
slope, intercept, r_value, p_value, std_error = stats.linregress(objectiveLoss[:-1], avgTime[1:])
print("Linear Regression on BO objective loss vs Flight Simulation Time had slope: ", slope, " intercept: ", intercept, " and r value of ", r_value)
print("")
print("Best performing PID values in terms of simulator fitness from cycle number ", c + 1 , ":")
print("")
print("Pitch:   Prop: ", PIDParameters[0][c], " Int: ", PIDParameters[1][c], " Deriv: ", PIDParameters[2][c])
print("Roll:    Prop: ", PIDParameters[3][c], " Int: ", PIDParameters[4][c], " Deriv: ", PIDParameters[5][c])
print("Yaw:     Prop: ", PIDParameters[6][c], " Int: ", PIDParameters[7][c], " Deriv: ", PIDParameters[8][c])
if POLICYMODE == 'HYBRID':
    print("YawRate: Prop: ", PIDParameters[9][c], " Int: ", PIDParameters[10][c], "Deriv: ", PIDParameters[11][c])
if POLICYMODE == 'RATE':
    print("PitchRt: Prop: ", PIDParameters[9][c], " Int: ", PIDParameters[10][c], " Deriv: ", PIDParameters[11][c])
    print("RollRate:Prop: ", PIDParameters[12][c], " Int: ", PIDParameters[13][c], " Deriv: ", PIDParameters[14][c])
    print("YawRate: Prop: ", PIDParameters[15][c], " Int: ", PIDParameters[16][c], "Deriv: ", PIDParameters[17][c])
