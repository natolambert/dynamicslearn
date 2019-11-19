import os
import sys
from dotmap import DotMap

import pandas as pd
import numpy as np
import torch
import math

from learn.control.pid import PID
from learn.control.pid import PidPolicy
from learn.utils.data import cwd_basedir
import logging
import hydra

log = logging.getLogger(__name__)
######################################################################
@hydra.main(config_path='conf/mpc.yaml')
def optimizer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")


if __name__ == '__main__':
    sys.exit(optimizer())
