#!/usr/bin/env python3
import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
from time import time, strftime, localtime
import logging
import hydra
import gym
import torch
import numpy as np
from learn.envs.model_env import ModelEnv


from learn.simulate_sac import SAC, ReplayBuffer, eval_mode, set_seed_everywhere, evaluate_policy, update_plot
from learn.utils.sim import *
from learn.trainer import train_model

def mbpo_experiment(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    real_env = gym.make(cfg.env.params.name)

    if cfg.metric.name == 'Living':
        metric = living_reward
    elif cfg.metric.name == 'Rotation':
        metric = rotation_mat
    elif cfg.metric.name == 'Square':
        metric = squ_cost
    else:
        raise ValueError("Improper metric name passed")

    # instantiate model for this round
    X = []
    U = []
    dX = []
    dynamics_model, train_log = train_model(X, U, dX, cfg.model)

    D_env = (X, U, dX)

    # loads a dataset and pretrains a model if instructed to do so
    if cfg.alg.pretrain:
        raise NotImplementedError
    # variables
    n_epochs = cfg.mbpo.num_epochs
    e_steps = cfg.mbpo.env_steps
    m_rollouts = cfg.mbpo.model_rollouts
    k_steps = cfg.mbpo.k_steps
    g_steps = cfg.mbpo.g_steps

    def get_steps(k_steps, epoch):
        if type(k_steps) == int:
            return k_steps
        start_val = k_steps[0]
        end_val = k_steps[1]
        start_epoch = k_steps[2]
        end_epoch = k_steps[3]
        if epoch < start_epoch:
            return int(start_val)
        elif epoch >= end_epoch:
            return int(end_val)
        else:
            return int((start_val + (end_val - start_val) * (epoch - start_epoch) / (end_epoch - start_epoch)))

    def parallel_rollout(model_env, D_env, replay_buffer, policy, m_rollouts, k):
        indexable = torch.stack(list(D_env.states0))
        num_samples = len(D_env)
        idx_state = np.random.randint(0, num_samples, m_rollouts)
        state_batch = torch.tensor(indexable[idx_state].cpu().numpy().squeeze())
        for i in range(k):
            action_batch = policy.sample_action_batch(state_batch).squeeze()
            next_state_batch, reward_batch, done_batch, _ = model_env.step_from(state_batch.to(cfg.device),
                                                                                action_batch)

            for s_b, a_b, r_b, ns_b, d_b in zip(state_batch, action_batch, reward_batch, next_state_batch,
                                                done_batch):
                replay_buffer.add(s_b, a_b, r_b, ns_b, d_b)

            state_batch = torch.tensor(next_state_batch)

        return replay_buffer

    model_env = ModelEnv(real_env, dynamics_model)

    obs_dim = cfg.env.state_size
    action_dim = cfg.env.action_size
    target_entropy_coef = 1
    batch_size = cfg.alg.params.batch_size  # 512
    discount = cfg.alg.trainer.discount  # .99
    tau = cfg.alg.trainer.tau  # .005
    policy_freq = cfg.alg.trainer.target_update_period  # 2
    replay_buffer_size = int(cfg.alg.replay_buffer_size)  # 1000000
    eval_freq = cfg.alg.params.eval_freq  # 10000
    num_eval_episodes = cfg.alg.params.num_eval_episodes  # 5
    num_eval_timesteps = cfg.alg.params.num_eval_timesteps  # 1000

    replay_buffer = ReplayBuffer(obs_dim, action_dim, cfg.device, replay_buffer_size)

    policy = SAC(cfg.device, obs_dim, action_dim,
                 hidden_dim=cfg.alg.layer_size,
                 hidden_depth=cfg.alg.num_layers,
                 initial_temperature=cfg.alg.trainer.initial_temp,
                 actor_lr=cfg.alg.trainer.actor_lr,  # 1E-3,
                 critic_lr=cfg.alg.trainer.critic_lr,  # 1E-3,
                 actor_beta=cfg.alg.trainer.actor_beta,  # 0.9,
                 critic_beta=cfg.alg.trainer.critic_beta,  # 0.9,
                 log_std_min=cfg.alg.trainer.log_std_min,  # -10,
                 log_std_max=cfg.alg.trainer.log_std_max)  # 2)

    step = 0
    saved_idx = 0

    target_entropy = -action_dim * target_entropy_coef

    to_plot_rewards = []
    rewards = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps, None)
    to_plot_rewards.append(rewards)

    layout = dict(
        title=f"Learning Curve Reward vs Number of Steps Trials (Env: {cfg.env.paramsname}, Alg: {cfg.alg.name})",
        xaxis={'title': f"Steps*{e_steps}"},
        yaxis={'title': f"Avg Reward Num:{num_eval_episodes}"},
        font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
        legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'}
    )
    lab = cfg.full_log_dir[cfg.full_log_dir.rfind('/') + 1:]

    set_seed_everywhere(cfg.random_seed)

    def get_recent_transitions(sasdata, num):
        if len(sasdata) <= num:
            return sasdata
        else:
            new_data = SASDataset()
            ind = len(sasdata)-1
            while len(new_data) < num:
                sars = sasdata[ind]
                new_data.add(sars)
                ind -= 1
            return new_data


    # each epoch is for evaluation of the policy  #### #### #### #### #### #### #### #### #### #### #### ####
    for n in range(n_epochs):
        if n % 100 == 0:
            log.info(f"Epoch {n}")

        # Train model
        if len(D_env) > 0:
            if cfg.mbpo.dynam_size > 0:
                D_train = get_recent_transitions(D_env, cfg.mbpo.dynam_size)
                # dynamics_model = train_model(D_train, dynamics_model, cfg, log)
                dynamics_model, train_log = train_model(X, U, dX, cfg.model)
            else:
                # dynamics_model = train_model(D_env, dynamics_model, cfg, log)
                dynamics_model, train_log = train_model(X, U, dX, cfg.model)

            model_env = ModelEnv(real_env, cfg, dynamics_model, metric)

        s_t = real_env.reset()
        e = 0
        # take e env steps each epoch #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        while e < e_steps:
            # log.info(f"===================================")
            if e % 100 == 0:
                log.info(f"- EnvStep {e}")
                # returns = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps, None)

            with torch.no_grad():
                with eval_mode(policy):
                    action = policy.sample_action(s_t)

            s_tp1, r, done, _ = real_env.step(action)
            D_env.add(SAS(torch.tensor(s_t, device='cuda', dtype=torch.float32),
                          torch.tensor(action, device='cuda', dtype=torch.float32),
                          torch.tensor(s_tp1, device='cuda', dtype=torch.float32)))
            s_t = s_tp1
            # perform m rollouts from current state #### #### #### #### #### #### #### #### #### #### ####
            replay_buffer = parallel_rollout(model_env, D_env, replay_buffer, policy, m_rollouts,
                                             get_steps(k_steps, n))

            # for m in range(m_rollouts):
            #     # if m % 100 == 0:
            #     #     log.info(f"- - Model R {m}")
            #     # sample and set starting state
            #     num = len(D_env)
            #     idx_state = np.random.randint(0, num, 1)
            #     s_t_m = D_env[idx_state].s0.cpu().numpy().squeeze()
            #     _ = model_env.reset()
            #     model_env.set_state(s_t_m)
            #
            #     # perform k step model rollout, append data #### #### #### #### #### #### #### #### #### ####
            #     for i in range(get_steps(k_steps, n)):
            #         action_m = policy.sample_action(s_t_m)  # policy.get_action(s_t)[0]
            #         s_tp1_m, r_m, done_m, _ = model_env.step(action_m)
            #
            #         D_model.add(SAS(torch.tensor(s_t_m, device='cuda', dtype=torch.float32),
            #                         torch.tensor(action_m, device='cuda', dtype=torch.float32),
            #                         torch.tensor(s_tp1_m, device='cuda', dtype=torch.float32)))
            #
            #         # add to buffer
            #         replay_buffer.add(s_t_m, action_m, r_m, s_tp1_m, done_m)
            #
            #         s_t_m = s_tp1_m

            # if e > 3: replay_buffer = parallel_rollout(model_env, D_env, replay_buffer, policy, m_rollouts, get_steps(k_steps, n))

            #### END K LOOP #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

            #### END M LOOP #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # udpate policy
            for g in range(g_steps):
                policy.update(
                    replay_buffer,
                    g,  # pass g here rather then the step in hopes to make the policy updates more stable
                    log,
                    batch_size,
                    discount,
                    tau,
                    policy_freq=2,
                    target_entropy=target_entropy)

            # if e % 100 == 0:
            #     returns = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps,
            #                               None)
            #     to_plot_rewards.append(returns)
            #     update_plot(to_plot_rewards, vis_alg, layout, lab)

            #### END G LOOP #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

            # if (e % 100) == 0:
            #     trial_log = dict(
            #         env_name=cfg.env.name,
            #         trial_num=saved_idx,
            #         replay_buffer=[],
            #         policy=policy,
            #         rewards=to_plot_rewards,
            #     )
            #     save_log(cfg, step, trial_log)
            #     saved_idx += 1
            e += 1

        #### END E LOOP #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # eval every epoch (n)?
        returns = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps,
                                  None)
        to_plot_rewards.append(returns)

        trial_log = dict(
            env_name=cfg.env.params.name,
            trial_num=saved_idx,
            replay_buffer=[],
            dynamics_model = dynamics_model if cfg.mbpo.save_model else [],
            policy=policy,
            rewards=to_plot_rewards,
        )
        save_log(cfg, saved_idx, trial_log)
        saved_idx += 1

def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


log = logging.getLogger(__name__)
@hydra.main(config_path='conf/config-base.yaml')
def experiment(cfg):
    mbpo_experiment(cfg)


if __name__ == '__main__':
    sys.exit(experiment())