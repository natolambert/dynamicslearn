

# dynamics-learn
Working directory for my work on model-based reinforcement learning for novel robots. Best for robots with high test cost and difficult to model dynamics. Contact: [nol@berkeley.edu](mailto:nol@berkeley.edu)
First paper website: [https://sites.google.com/berkeley.edu/mbrl-quadrotor/](https://sites.google.com/berkeley.edu/mbrl-quadrotor/)
There is current future work using this library, such as attempting to control the Ionocraft with model-based RL.  [https://sites.google.com/berkeley.edu/mbrl-ionocraft/](https://sites.google.com/berkeley.edu/mbrl-ionocraft/)

This directory is working towards an implementation of many simulated model-based approaches on real robots. For current state of the art in simulation, see this work from Prof Sergey Levine's group: [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114).

Future implementations work towards controlled flight of the ionocraft,
[with a recent publication in Robotics and Automation Letters](https://ieeexplore.ieee.org/document/8373697/)
and in the future for transfer learning of dynamics on the Crazyflie 2.0 Platform.

Some potentially noteable implementations include:
- probablistic nueral network in pytorch
- gaussian loss function for said pytorch probablistic neural network
- random shooting MPC implementation with customizable cost / reward function (See cousin repo: https://github.com/natolambert/ros-crazyflie-mbrl)

Usage is generally of the form, with hydra enabling more options:
```
$ python learn/trainer.py robot=iono
```

Main Scripts:
---------------
- `learn/trainer.py`: is for training dynamics models (P,PE,D,DE) on experimental data. The training process uses [Hydra](https://github.com/facebookresearch/hydra) to allow easy configuration of which states are used and how the predictions are formatted. 
- `learn/plot.py`: a script for viewing different types of predictions, under improvement

In Development:
-------------
- `learn/bo_pid.py`: For generating PID parameters using a dynamics model as a simulation environment. This will eventually extend beyond PID control. See the controllers directory `learn/control`. I am working to integrate [opto](https://github.com/robertocalandra/opto).
- `learn/pipps_experiment.py`: A reimplementation of the paper ["PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos"](https://arxiv.org/abs/1902.01240). I wrote a blog post summarizing the main derivation behind this work [here](https://medium.com/me/stats/post/4546434c84b0).

Related Code for Experiments:
-----------------------------
CF Firmware: https://github.com/natolambert/crazyflie-firmware-pwm-control
  - forked from: https://github.com/bitcraze/crazyflie-firmware

Ros code: https://github.com/natolambert/ros-crazyflie-mbrl
  - uh from: https://github.com/whoenig/crazyflie_ros
  - want to use: https://github.com/USC-ACTLab/crazyswarm
