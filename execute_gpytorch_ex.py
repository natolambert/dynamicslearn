# Notes for trying gpytorch example

"""
Our problem is iterating of data which is PID parameters (x) and 
  the output is some metric of the simualted PID pereformance (y).
  My first idea for PID performance is the integral of absolute val the euler angles pitch and roll over time.
  The integral will encourage the PID to travel to 0 and stay there across the time horizon of prediction T

There's really two options for PID generation - 
1. PID Tuning similar to PILCO as in: Model-Based Policy Search for Automatic Tuning of Multivariate PID Controllers
2. PID Tuning with Bayesian Opt similar to: Goal-Driven Dynamics Learning via Bayesian Optimization
"""

