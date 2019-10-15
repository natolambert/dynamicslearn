import os
import sys

sys.path.append(os.getcwd())

from learn.control.pipps_policygrad import PIPPS_policy
import gym
from learn.model_general_nn import GeneralNN

from learn.model_ensemble_nn import EnsembleNN

from learn.utils.nn import *
from learn.utils.data import *
# import utils.rl

# saving files

import logging
import hydra

log = logging.getLogger(__name__)


def train_model(X, U, dX, model_cfg):
    log.info("Training Model")
    dx = np.shape(X)[1]
    du = np.shape(U)[1]
    dt = np.shape(dX)[1]

    # if set dimensions, double check them here
    if model_cfg.training.dx != -1:
        assert model_cfg.training.dx == dx, "model dimensions in cfg do not match data given"
    if model_cfg.training.du != -1:
        assert model_cfg.training.dx == du, "model dimensions in cfg do not match data given"
    if model_cfg.training.dt != -1:
        assert model_cfg.training.dx == dt, "model dimensions in cfg do not match data given"

    train_log = dict()
    nn_params = {  # all should be pretty self-explanatory
        'dx': dx,
        'du': du,
        'dt': dt,
        'hid_width': model_cfg.training.hid_width,
        'hid_depth': model_cfg.training.hid_depth,
        'bayesian_flag': model_cfg.training.probl,
        'activation': Swish(),  # TODO use hydra.utils.instantiate
        'dropout': model_cfg.training.extra.dropout,
        'split_flag': False,
        'ensemble': model_cfg.ensemble
    }

    train_params = {
        'epochs': model_cfg.optimizer.epochs,
        'batch_size': model_cfg.optimizer.batch,
        'optim': model_cfg.optimizer.name,
        'split': model_cfg.optimizer.split,
        'lr': model_cfg.optimizer.lr,  # bayesian .00175, mse:  .0001
        'lr_schedule': model_cfg.optimizer.lr_schedule,
        'test_loss_fnc': [],
        'preprocess': model_cfg.optimizer.preprocess,
    }

    train_log['nn_params'] = nn_params
    train_log['train_params'] = train_params

    if model_cfg.ensemble:
        newNN = EnsembleNN(nn_params, model_cfg.training.E)
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

    else:
        newNN = GeneralNN(nn_params)
        newNN.init_weights_orth()
        if nn_params['bayesian_flag']: newNN.init_loss_fnc(dX, l_mean=1, l_cov=1)  # data for std,
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

    if model_cfg.ensemble:
        min_err = np.min(acctrain, 0)
        min_err_test = np.min(acctest, 0)
    else:
        min_err = np.min(acctrain)
        min_err_test = np.min(acctest)

    train_log['testerror'] = acctest
    train_log['trainerror'] = acctrain
    train_log['min_trainerror'] = min_err
    train_log['min_testerror'] = min_err_test

    return newNN, train_log

@hydra.main(config_path='conf/pipps.yaml')
def run_pipps(cfg):
    raise NotImplementedError("Not totally transitioned to new code")
    ############ SETUP EXPERIMENT ENV ########################
    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make("CartPoleContEnv-v0")

    observations = []
    actions = []
    rewards_tot = []
    rand_ep = 100
    for i_episode in range(rand_ep):
        observation = env.reset()
        rewards = []
        for t in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            # reshape obersvation
            observation = observation.reshape(-1, 1)
            # print(action)
            observations.append(observation)
            actions.append([action])
            rewards.append(reward)

            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                # print("Ep reward: ", np.sum(rewards))
                rewards_tot.append(np.sum(rewards))
                break

    ############ PROCESS DATA ########################
    o = np.array(observations).squeeze()  # cartpole

    actions = np.array(actions).reshape(-1, 1)
    # shape into trainable set
    d_o = o[1:, :] - o[:-1, :]
    actions = actions[:-1, :]
    o = o[:-1, :]

    print('---')
    print("X has shape: ", np.shape(o))
    print("U has shape: ", np.shape(actions))
    print("dX has shape: ", np.shape(d_o))
    print("Mean Random Reward: ", np.mean(rewards_tot))
    print('---')

    ############ TRAIN MODEL ########################
    model, train_log = train_model(o, actions, d_o, cfg.model)
    # model.store_training_lists(list(data['states'].columns),
    #                            list(data['inputs'].columns),
    #                            list(data['targets'].columns))

    msg = "Trained Model..."
    # msg += "Prediction List" + str(list(data['targets'].columns)) + "\n"
    msg += "Min test error: " + str(train_log['min_testerror']) + "\n"
    msg += "Mean Min test error: " + str(np.mean(train_log['min_testerror'])) + "\n"
    msg += "Min train error: " + str(train_log['min_trainerror']) + "\n"
    log.info(msg)

    ######### PIPPS PART ###########

    pipps_nn_params = {  # all should be pretty self-explanatory
        'dx': np.shape(o)[1],
        'du': np.shape(actions)[1],
        'hid_width': 100,
        'hid_depth': 2,
        'bayesian_flag': True,
        'activation': Swish(),
        'dropout': 0.2,
        'bayesian_flag': False
    }

    # for the pipps policy update step
    policy_update_params = {
        'P': 20,
        'T': 25,
        'learning_rate': 3e-4,
    }

    ############ INITAL PIPPS POLICY ########################
    # init PIPPS policy
    PIPPSy = PIPPS_policy(pipps_nn_params, policy_update_params, newNN)
    PIPPSy.init_weights_orth()  # to see if this helps initial rollouts
    # set cost function
    """
    Observation:
            Type: Box(4)
            Num	Observation             Min             Max
            0	Cart Position           - 4.8            4.8
            1	Cart Velocity           - Inf            Inf
            2	Pole Angle              - 24 deg        24 deg
            3	Pole Velocity At Tip    - Inf            Inf
    """

    def simple_cost_cartpole(vect):
        l_pos = 100
        l_vel = 10
        l_theta = 200
        l_theta_dot = 5
        return 1 * (l_pos * (vect[0] ** 2) + l_vel * vect[1] ** 2 \
                    + l_theta * (vect[2] ** 2) + l_theta_dot * vect[3] ** 2)

    def simple_cost_car(vect):
        l_pos = 10
        l_vel = 2.5
        return -l_pos * vect[0]  # + l_vel*vect[1]**2

    PIPPSy.set_cost_function(simple_cost_cartpole)

    # set baseline function
    PIPPSy.set_baseline_function(np.mean)
    PIPPSy.set_statespace_normal([], [])

    PIPPSy.policy_step(o)

    ############ PIPPS ITERATIONS ########################
    P_rollouts = 20
    for p in range(P_rollouts):
        print("------ PIPPS Training Rollout", p, " ------")
        observations_new = []
        actions = []
        rewards_fin = []
        for i_episode in range(15):
            observation = env.reset()
            rewards = []
            for t in range(100):
                if p > 10: env.render()
                observation = observation.reshape(-1)
                # print(observation)
                # action = PIPPSy.predict(observation)  # env.action_space.sample()
                # env.action_space.sample()
                action = PIPPSy.forward(torch.Tensor(
                    [observation]), normalize=False)
                # PIPPSy.viz_comp_graph(action.requires_grad_(True))
                # print(action)
                # action = action.int().data.numpy()[0]
                action = torch.clamp(action, min=-1, max=1).data.numpy()
                # print(action)

                observation, reward, done, info = env.step(action[0])

                observation = observation.reshape(-1, 1)
                observations_new.append(observation)
                actions.append([action])
                rewards.append(reward)

                if done:
                    # print("Episode finished after {} timesteps".format(t+1))
                    # print(np.sum(rewards))
                    rewards_fin.append(np.sum(rewards))
                    break
        observations_new = np.array(observations_new).squeeze()
        o = np.concatenate((o, observations_new), 0)
        print("New Dataset has shape: ", np.shape(o))
        print("Reward at this iteration: ", np.mean(rewards_fin))
        print('---')
        PIPPSy.policy_step(np.array(o))
        PIPPSy.T += 1  # longer horizon with more data
        # print('---')

if __name__ == '__main__':
    sys.exit(run_pipps())