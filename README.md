# dynamics-learn
Working directory for my final project in EECS289a: Intro to Machine Learning and EE291e: Hybrid Systems and Intelligent Control.

-------------

# Project Goal
This project is exploring the functionality of different machine learning structures to model unknown dynamics. In simulation, we can rapidly collect data and test on different pre-made dynamics models. This code-base is meant to serve as the foundation for testing in experiment. Below is an example of an ionocraft hovering off of learned dynamics and model predictive control.

![Alt Text](https://github.com/natolambert/dynamics-learn/blob/master/ex.gif)


# Desired Workflow: 
The plan would be for a user to come in with a dynamics files in the structure of the two given, intitialize a controller of the class given, then explore the environment and gather dynamics data. With this data the user will be able to try multiple learning techniques to model the dynamics. Once a sufficient model of dynamics is acquired, the user can simulate trajectories on controllers that only know the learned dynamics, which is a model-based controller.

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
