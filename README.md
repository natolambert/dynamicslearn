# dynamics-learn
Working directory for my final project in EECS289a: Intro to Machine Learning and EE291e: Hybrid Systems and Intelligent Control.

-------------

# Project Goal
This project is exploring the functionality of different machine learning structures to model unknown dynamics. In simulation, we can rapidly collect data and test on different pre-made dynamics models. This code-base is meant to serve as the foundation for testing in experiment.

# Implemented:
- set up dynamics exploration system around an equilibrium point to colelct training data
- dynamics model for quadrotor
- methods to create dynamics estimators
  - least squares

# To Do:
- visualization program of dynamics exploration
  - neural networks
  - gaussian processes
  - advanced?
- dynamcis model for any other robot to consider
- writing up all above
- control structure
  - mpc
  - iLQR
  - many choices
  
--------------
 
# Papers Read in Rough Order:
 1. Nagabandi - Neural Network Dynamics Models for Control of Under-actuated Legged Millirobots
 2. Bansal - Learning Quadrotor Dynamics Using Neural Network for Flight Control
 3. Deisenroth - PILCO
 4. Bansal - Goal-Driven Dynamics... 
 5. Chua - On the Importance of Uncertainty.... 
