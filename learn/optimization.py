from dotmap import DotMap
import numpy as np

import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression

class Optimize(OptTask):
    def __init__(self, opt_cfg):
        self.cfg = opt_cfg
        self.p = DotMap()
        self.p.verbosity = opt_cfg.verbosity
        self.p.acq_func = EI(model=None, logs=None)
        self.p.model = regression.GP
        self.opt = opto.BO(parameters=p, task=self.task, stopCriteria=self.Stop)

    def optimize(self):
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