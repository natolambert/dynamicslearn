# dynamics-learn
Working directory for my work on model-based reinforcement learning for novel robots. Best for robots with high test cost and difficult to model dynamics. Contact: [nol@berkeley.edu](mailto:nol@berkeley.edu)

This directory is working towards an implementation of many simulated model-based approaches on real robots. For current state of the art in simulation, see this work from Prof Sergey Levine's group: [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114).

Future implementations work towards controlled flight of the ionocraft,
[with a recent publication in Robotics and Automation Letters](https://ieeexplore.ieee.org/document/8373697/)
and in the future for transfer learning of dynamics on the Crazyflie 2.0 Platform.

Some potentially noteable implementations include:
- probablistic nueral network in pytorch
- gaussian loss function for said pytorch probablistic neural network
- random shooting MPC implementation with customizable cost / reward function

Literature Breakdown:
---------------------

For current state of the art, as I said see [K. Chua et al.](https://arxiv.org/abs/1805.12114). This paper covers the design choices between deterministic and probablistic neural networks for learning, along with a discussion of ensemble learning. It then covers a new MPC technique needed for higher state systems coined Trajectory Sampling. Especially for our goal of implementing this on real robots, some other recent papers that cover their own implementations can prove more useful, such as [Bansal et al. learning trajectories on the CrazyFlie](https://ieeexplore.ieee.org/document/7798978/) or [Nagabundi et al. with millirobots](https://arxiv.org/abs/1708.02596). A more theoretical framework for model-based learning includes the [PILCO Algorithm](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf) and .... will update with what I feel is relevant.

For some general reinforcement learning references, see a [lecture series by Deepmind's David 
Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), [the Deep RL Bootcamp 2017 
Lectures](https://sites.google.com/view/deep-rl-bootcamp/lectures) from various 
researchers, the [Deep Learning Book from Goodfellow & 
MIT](https://www.deeplearningbook.org/), or [Berkeley's own Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)(that I will be taking in the fall).

File Breakdown
--------------
- controllers.py: Location for model predictive control, other controllers, and objective functions
- dynamics.py: Base class for dynamics and tools for simulating data
- /dynamics_files: Location for robot dynamics files
- loss_pnn_gaussian.py: Loss function for PNN
- model.py: Deterministic neural net, location for other models to try such as linear regression or gaussian processes
- model_general_nn.py: NN object that can operate in probablistic or deterministic fasion. Configurable for prediction type
- script_experiement.py: Script for controllign robot via serial communication
- script_testing.py: Internal code testing, good walkthrough for figuring out how pieces work together
- utils_plot.py: Visualization files
- utils_data.py: Data processing files

Features to add:
- easier workflow with documentation
- ensemble NN and methods for reducing variances
- more plot & data utilities. May change dynamics data to pandas frames
