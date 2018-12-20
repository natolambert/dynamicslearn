# dynamics-learn
Working directory for my work on model-based reinforcement learning for novel robots. Best for robots with high test cost and difficult to model dynamics. Contact: [nol@berkeley.edu](mailto:nol@berkeley.edu)
Project website: [https://sites.google.com/berkeley.edu/mbrl-quadrotor/](https://sites.google.com/berkeley.edu/mbrl-quadrotor/)

This directory is working towards an implementation of many simulated model-based approaches on real robots. For current state of the art in simulation, see this work from Prof Sergey Levine's group: [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114).

Future implementations work towards controlled flight of the ionocraft,
[with a recent publication in Robotics and Automation Letters](https://ieeexplore.ieee.org/document/8373697/)
and in the future for transfer learning of dynamics on the Crazyflie 2.0 Platform.

Some potentially noteable implementations include:
- probablistic nueral network in pytorch
- gaussian loss function for said pytorch probablistic neural network
- random shooting MPC implementation with customizable cost / reward function (See cousin repo: https://github.com/natolambert/ros-crazyflie-mbrl)

File Breakdown:
---------------
- utils/ holds utility files for reading data (reach out for dataset to collaborate on), plotting tools, and some nn functions that were written before being added to pytorch.
- execute_gen_PID and execute_train_nn are used to generate either a PID or a dynamics model from data. Expect another to be added for model free policies.
- gymenv's take a dynamics model and provide a wrapper for it to be used as a standard gym environement.
- model_'s are our various implementations of different dynamics models and policies.
- pid contains the PID class, not currently used.
- plot_'s contain plotting code used for papers, could be a good reference for plotting dynamics model function.
- policy_'s contain implementations of different state of the art model free algorithms in coordination with rlkit (), but there is a strong chance with branch from this.

Requirements:
-------------
TBD, currently code runs on pytorch 0.4.1, but we wish to update to 1.0.

Future Implementations:
---------------------

Feel free to request feature additions if you are interested in the line of work, but things in our pipeline:
- Implementation of regularization term for training policies on a learned neural network dynamics model, as introduced here: http://proceedings.mlr.press/v80/parmas18a.html
- Additional tools for training dynamics models on two step predictions, improved ensemble implementation, and simpler models
- PID tuning with gaussian processes

Literature Breakdown:
---------------------

For current state of the art, as I said see [K. Chua et al.](https://arxiv.org/abs/1805.12114). This paper covers the design choices between deterministic and probablistic neural networks for learning, along with a discussion of ensemble learning. It then covers a new MPC technique needed for higher state systems coined Trajectory Sampling. Especially for our goal of implementing this on real robots, some other recent papers that cover their own implementations can prove more useful, such as [Bansal et al. learning trajectories on the CrazyFlie](https://ieeexplore.ieee.org/document/7798978/) or [Nagabundi et al. with millirobots](https://arxiv.org/abs/1708.02596). A more theoretical framework for model-based learning includes the [PILCO Algorithm](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf) and .... will update with what I feel is relevant.

For some general reinforcement learning references, see a [lecture series by Deepmind's David 
Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), [the Deep RL Bootcamp 2017 
Lectures](https://sites.google.com/view/deep-rl-bootcamp/lectures) from various 
researchers, the [Deep Learning Book from Goodfellow & 
MIT](https://www.deeplearningbook.org/), or [Berkeley's own Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)
