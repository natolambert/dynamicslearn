'''Functions to import initial conditions or parameters'''

'''Reads from excel file the optimal nn_params and train_params. Returns two respective lists '''
from ExecuteTrain import getNewNNParams, getNewTrainParams
from collections import OrderedDict
import xlrd
import ctypes
import numpy as np

def getOptimalNNTrainParams(includePosition, nState, nInput, stack, lr = None, epochs = None):
    ENSEMBLES = getNewNNParams(nState, nInput, stack)['ensemble']
    dataPath = "ParameterSweep4_Favoritism.xlsx"
    workbook = xlrd.open_workbook(dataPath)
    sheet = workbook.sheet_by_index(0)
    nn_params = []
    train_params = []

    for i in range(ENSEMBLES):
        row = [cell.value for idx, cell in enumerate(sheet.row(i + 1))]
        row = row[:-2] #don't include minimum loss and the key
        eNNParams = getNewNNParams(nState, nInput, stack)
        eTrainParams = getNewTrainParams()
        eNNParams['dropout'] = row[0]
        eTrainParams['lr'] = lr or row[1]
        eTrainParams['lr_schedule'] = [row[2], row[3]]
        eTrainParams['batch_size'] = int(row[4])
        eTrainParams['epochs'] = epochs or int(row[5]) #chaned to 5 since cycle is running
        numStack = eNNParams['stack']
        if includePosition:
            eNNParams['dx'] += 3 * numStack
            eNNParams['dt'] += 3
        nn_params.append(eNNParams)
        train_params.append(eTrainParams)
    return nn_params, train_params

###############################################################################
'''FOR SIMULATION'''

def degToRad(states):
    '''Takes in state(s) and converts euler angles and omegas from degrees to radians'''
    angles = states[:, 6:9]
    omegas = states[:, 9:12]
    states[:,6:9] = np.radians(angles)
    states[:,9:12] = np.radians(omegas)
    return states

def radToDeg(states):
    '''Takes in state(s) and converts euler angles and omegas from radians to degrees'''
    angles = states[:,6:9]
    omegas = states[:, 9:12]
    states[:,6:9] = np.degrees(angles)
    states[:,9:12] = np.degrees(omegas)
    return states

def getState():
    '''Gets a random state (including position) as an initial condition for testing policy in simulations'''
    size = 100
    xy = np.random.uniform(-.2, .2, (size, 2))
    z = np.random.uniform(10, 10, (size, 1))
    #for the velocities, make them range <.1 as well
    v = np.random.uniform(-.3, .3, (size, 3))
    #euler angles and velocities can be random with values <.3
    #yaw = np.random.uniform(-1, 1,(size, 1))
    angles = np.random.uniform(-.26, .26, (size, 3))
    omegas = np.random.uniform(-.02,.02,(size, 3))
    X = np.hstack((xy, z, v, angles, omegas))
    return X
