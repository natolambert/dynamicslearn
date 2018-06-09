# dynamics-learn
Working directory for my work on model-based reinforcement learning for novel robots. Best for robots with high test cost and difficult to model dynamics. Contact: [nol@berkeley.edu](mailto:nol@berkeley.edu)

This directory is working towards an implementation of many simulated model-based approaches on real robots. For current state of the art in simulation see this work from Prof Sergey Levine's group: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models,  https://arxiv.org/abs/1805.12114.

Future implementations work towards controlled flight of the ionocraft,
[with a recent publication in Robotics and Automation Letters](https://ieeexplore.ieee.org/document/8373697/)
and in the future for transfer learning of dynamcis on the Crazyflie 2.0 Platform.

Some potentially noteable implementations include:
- probablistic nueral network in pytorch
- gaussian loss function for said pytorch probablistic neural network
- random shooting MPC implementation with customizable cost / reward function
