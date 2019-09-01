'''Class analyzes trends in a given dataset and returns additional data based on rewards we want to acknowledge

Time: reward rollouts that had a long flight time above a certain threshold
SSE: reward rollouts that had minimal steady state error
Settling time: reward rollouts that had a quick settling time
Max Overshoot: reward rollouts that had a small max overshoot within the first few iterations
'''
import numpy as np
import math
import torch

class Analysis():
    def __init__(self, aDict):
        self.stack = aDict['stack']
        self.time = aDict['rewardTime']
        self.sse = aDict['rewardSSE']
        self.settle = aDict['rewardSettle']
        self.overshoot = aDict['rewardOver']
        self.dataOriginal = aDict['data']
        self.settleSeconds = aDict['SETTLESECONDS']
        self.settleBenchmark = aDict['SETTLEBENCHMARK']
        self.maxAngle = aDict['MAXANGLE']
        self.minFrames = aDict['MINIMUMFRAMES']
        self.extra = None
        self.tick = aDict['dt']
        self.dimX = aDict['dX']#dimensions of our original dataset states, actions, and resulting state
        self.dimU = aDict['dU']
        self.dimT = aDict['dT']

        #data
        self.X = self.dataOriginal[:,0 : self.dimX]
        self.U = self.dataOriginal[:,self.dimX:self.dimX+self.dimU]
        self.dX = self.dataOriginal[:,self.dimX+self.dimU : self.dimX+self.dimU+self.dimT] #don't want to take into account the REWARD just yet
        self.rewards = self.dataOriginal[:, self.dimX+self.dimU+self.dimT : self.dimX+self.dimU+self.dimT + 1]
        self.flightnum = self.dataOriginal[:, self.dimX+self.dimU+self.dimT+1: self.dimX +self.dimU+self.dimT+2]

        #getting the sizes and offsets of each of the flights in the dataset for easy access later
        index = 0
        self.offsets = []
        self.sizes = []
        length = np.shape(self.rewards)[1]
        slices = []
        while index < length:
            self.offsets += [int(index)]
            self.sizes  += [int(self.rewards[index, 0])]
            index += self.rewards[index, 0]

    def stackData(self, dataset):
        #stacks the dataset into self.stack stacks (states and inputs are stacked...resulting states are not)
        rows = np.shape(dataset)[0] - self.stack
        columns = self.dimX * self.stack + self.dimU * self.stack + self.dimX
        result = np.zeros((1, columns)) #dummy
        start = 0
        X = dataset[:, :self.dimX]
        U = dataset[:, self.dimX:self.dimX + self.dimU]
        dX = dataset[:, self.dimX + self.dimU: self.dimX + self.dimU +self.dimX]
        for i in range(rows): #stacking the data
            if self.flightnum[i, 0] != self.flightnum[i + (self.stack - 1), 0]:
                continue #this means the two concatenated datapoints are not from the same rollout
            currX = X[i, :]
            currU = U[i, :]
            currdX = dX[i + (self.stack - 1), :] #getting the resulting state after all the stacked inputs and states
            for j in range(self.stack - 1): #getting the subsequent states and inputs
                currX = np.hstack((currX, X[i + j + 1, :]))
                currU = np.hstack((currU, U[i + j + 1, :]))
            result = np.vstack((result, np.hstack((currX, currU, currdX))))
        X = result[1:, :self.dimX * self.stack]
        U = result[1:, self.dimX * self.stack: self.dimX*self.stack + self.dimU * self.stack]
        dX = result[1:, self.dimX* self.stack + self.dimU * self.stack:]
        return (X,U,dX)

    def stackOriginal(self): #stacks the original dataset passed into the class
        return self.stackData(self.dataOriginal)

    def stackThree(self, dataOne, dataTwo):
        #takes in two sets of triple tuples and vertically stacks each one
        return (np.vstack((dataOne[0], dataTwo[0])), np.vstack((dataOne[1], dataTwo[1])), np.vstack((dataOne[2], dataTwo[2])))

    def extraData(self):
        #returns the extra data appended by all the "rewards" we try to acknowledge
        result = (0,0) #dummy
        self.findSettle()
        print("")
        print("Rewarding rollouts:")
        if self.time:
            extra = self.rewardTime()
            if len(extra) > 1:
                result = extra if len(result) == 2 else self.stackThree(result, extra)
        if self.sse:
            extra = self.rewardSSE()
            if len(extra) > 1:
                result = extra if len(result) == 2 else self.stackThree(result, extra)
        if self.settle:
            extra = self.rewardSettle()
            if len(extra) > 1:
                result = extra if len(result) == 2 else self.stackThree(result, extra)
        if self.overshoot:
            extra = self.rewardOvershoot()
            if len(extra) > 1:
                result = extra if len(result) == 2 else self.stackThree(result, extra)
        self.extra = result
        if len(result) == 2:
            return (None,None)
        print("")
        return result

    def rewardTime(self):
        #use the dataset passed into the class and return a new STACKED dataset that consists of only the extra data we would want to append
        slices = []
        extra = np.zeros((1, self.dimX * (self.stack) + self.dimU * self.stack + self.dimT)) #dummy
        for i, s in enumerate(self.sizes):
            if s < self.minFrames:
                continue
            slices += [[self.offsets[i], self.offsets[i] + self.sizes[i]]]
        for slice in slices:
            extraData = self.stackData(self.dataOriginal[slice[0]:slice[1], :])
            extraData = np.hstack(extraData)
            extra = np.vstack((extra, extraData))
        if extra.shape[0] == 1:
            print("No data rewarded for flight time.")
            return [None] #dummy
        print(extra.shape[0] - 1, " datapoints rewarded for flight time.")
        endStates = self.dimX*self.stack
        endInputs = self.dimU*self.stack
        endOutputs = self.dimT
        return (extra[:, :endStates], extra[:, endStates: endStates + endInputs], extra[: ,endStates + endInputs: endStates + endInputs + endOutputs])

    def rewardSettle(self):
        #things to check:
        extra = np.zeros((1, self.dimX * (self.stack) + self.dimU * self.stack + self.dimT)) #dummy
        for i in range(len(self.settleTimes)):
            if self.settleTimes[i] != None and self.settleTimes[i] <= self.settleBenchmark/self.tick: #meaning we had a settle time that was noteworthy!
                start = self.offsets[i]
                size = self.settleTimes[i] #let's only reward the time leading up to settle time...leave the time afterwards for SSE
                extra = np.vstack((extra, np.hstack(self.stackData(self.dataOriginal[start: start + size, :]))))
        if extra.shape[0] == 1:
            print("No data rewarded for settle time.")
            return [None] #dummy
        print(extra.shape[0] - 1, " datapoints rewarded for settle time.")
        endStates = self.dimX*self.stack
        endInputs = self.dimU*self.stack
        endOutputs = self.dimT
        return (extra[:, :endStates], extra[:, endStates: endStates + endInputs], extra[: ,endStates + endInputs: endStates + endInputs + endOutputs])

    def findSettle(self):
        #helper function to find the settle time of a rollout. Basicallly, if it remains within a certain boundary for an acceptably long time
        self.settleTimes = []
        for i in range(len(self.offsets)): #offsets and sizes are the same length
            size = self.sizes[i]
            start = self.offsets[i]
            curr = start
            setTime = None
            setCount = 0
            while curr < start + size:
                if self.settledState(self.dataOriginal[curr, :]): #if it is considered steady, increment
                    setCount += 1
                else:
                    setCount = 0
                if setCount >= self.settleSeconds/self.tick: #otherwise, check if we have passed the minimum amount of ticks to consider it settled
                    setTime = curr - start
                    break
                curr += 1
            self.settleTimes += [setTime]

    def settledState(self, row):
        #checking if a specific state is considered "settled" (if it's within a certain boundary)
        yaw = 1 if abs(row[6]) < (self.maxAngle) else 0
        pitch = 1 if abs(row[7]) < (self.maxAngle) else 0
        roll = 1 if abs(row[8]) < (self.maxAngle) else 0
        return yaw + pitch + roll >= 2

    def rewardSSE(self):
        #we would need to check the settle time. Then, check the total error of all the euler angles after that time.
        SSE = [] #stores the average error after settle time for each rollout...contains None if there was no settle time
        extra = np.zeros((1, self.dimX * (self.stack) + self.dimU * self.stack + self.dimT)) #dummy
        for i in range(len(self.offsets)):
            setTime = self.settleTimes[i]
            if setTime == None:
                SSE += [None]
                continue
            start = self.offsets[i]
            size = self.sizes[i]
            subset = self.dataOriginal[start + setTime: start + size, 6:9] #just the euler angles of the states after settling
            avg = np.sum(subset)/subset.shape[0]
            SSE += [avg]
        for i in range(len(SSE)):
            erroravg = SSE[i]
            if erroravg == None:
                continue
            if erroravg < self.maxAngle:
                start = self.offsets[i] + self.settleTimes[i] #start only after settleTime...leave stuff before for settleTimerewards
                end = self.offsets[i] + self.sizes[i]
                extra = np.vstack((extra, np.hstack(self.stackData(self.dataOriginal[start: end, :]))))
        if extra.shape[0] == 1:
            print("No datapoints rewarded for SSE.")
            return [None] #dummy
        print(extra.shape[0] - 1, " datapoints rewarded for SSE.")
        endStates = self.dimX*self.stack
        endInputs = self.dimU*self.stack
        endOutputs = self.dimT
        return (extra[:, :endStates], extra[:, endStates: endStates + endInputs], extra[: ,endStates + endInputs: endStates + endInputs + endOutputs])

    def rewardOvershoot(self):
        #checking the max angle reached before settling down 
        result = []
        for i in range(len(self.settleTimes)):
            start = self.offsets[i]
            setTime = self.settleTimes[i]
            if setTime == None:
                result += [None]
                continue
            subset = self.dataOriginal[start:setTime, 6:9]
            maximum = np.amax(subset)
            result += [maximum]
        #just reward the one that has the smallest maximum overshoot
        if not any(result):
            print("No datapoints rewarded for overshoot.")
            return [None] #dummy
        idx = result.index(min([x for x in result if x is not None]))
        start = self.offsets[idx]
        end = start + self.settleTimes[idx]
        extra = np.hstack(self.stackData(self.dataOriginal[start:end, :]))
        print(extra.shape[0], " datapoints rewarded for overshoot.")
        endStates = self.dimX*self.stack
        endInputs = self.dimU*self.stack
        endOutputs = self.dimT
        return (extra[:, :endStates], extra[:, endStates: endStates + endInputs], extra[: ,endStates + endInputs: endStates + endInputs + endOutputs])
