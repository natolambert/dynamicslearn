#!/usr/bin/env python3
import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import hydra
import gym
import random

from learn import envs
from learn.utils.plotly import plot_rewards_over_trials, plot_rollout
from learn.utils.sim import *

LOG_FREQ = 10000
OUT_SIZE = 29


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, device, capacity):
        self.device = device
        self.capacity = capacity

        if type(obs_dim) == int:
            self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
            self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_policy(env, policy, step, L, num_episodes, num_eval_timesteps, video_dir=None, metric=None, show=False):
    returns = []
    start = time.time()
    for i in range(num_episodes):
        # print(f"Eval episode: {i}...")
        # video = VideoRecorder(env, enabled=video_dir is not None and i == 0)
        s = 0
        states = []
        actions = []
        obs = env.reset()
        done = False
        total_reward = 0
        while (not done) and (s < num_eval_timesteps):
            with torch.no_grad():
                with eval_mode(policy):
                    action = policy.select_action(obs)

            action_scale = env.action_space.high * (action + 1) / 2
            obs, reward, done, _ = env.step(action_scale)
            states.append(obs)
            actions.append(action_scale)
            if metric is not None:
                reward = metric(obs, action)
            # video.record()
            total_reward += reward
            s += 1
        returns.append(total_reward) #/len(states))

        if show:
            plot_rollout(states, actions, pry=[1, 0, 2])
    end = time.time()
    print(f"Rollout in {end - start} s, logged {len(states)}")
    L.info(f" - - Evaluated, mean reward {np.mean(returns)}, n={num_episodes}")
    return returns


def select_action(Actor, observation, cfg):
    with torch.no_grad():
        observation = torch.FloatTensor(observation).to(cfg.device)
        observation = observation.unsqueeze(0)
        if Actor.log_std_min is not None:
            mu, _, _, _ = Actor(
                observation, compute_pi=False, compute_log_pi=False)
        else:
            mu = Actor(
                observation, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()


def sample_action(Actor, observation, cfg):
    with torch.no_grad():
        observation = torch.FloatTensor(observation).to(cfg.device)
        observation = observation.unsqueeze(0)
        if Actor.log_std_min is not None:
            mu, pi, _, _ = Actor(observation, compute_log_pi=False)
        else:
            raise ValueError("Deterministic Actor has no Sampling Method")
        return pi.cpu().data.numpy().flatten()


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def gaussian_likelihood(noise, log_std):
    pre_sum = -0.5 * noise.pow(2) - log_std
    return pre_sum.sum(
        -1, keepdim=True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def apply_squashing_func(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def tie_weights(src, trg):
    assert type(src) == type(trg)

    trg.weight = src.weight
    trg.bias = src.bias


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_depth, log_std_min,
                 log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if hidden_depth == 2:
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_dim))
        elif hidden_depth == 3:
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_dim))
        elif hidden_depth == 4:
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_dim))
        else:
            raise ValueError(f"Actor Depth {hidden_depth} no supported")

        self.apply(weight_init)

    def forward(self,
                observation,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):

        mu, log_std = self.trunk(observation).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
            log_det = log_std.sum(dim=-1)
            entropy = 0.5 * (1.0 + math.log(2 * math.pi) + log_det)
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_likelihood(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)

        return mu, pi, log_pi, entropy


class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        assert state.size(0) == action.size(0)

        state_action = torch.cat([state, action], dim=1)
        return self.trunk(state_action)


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim):
        super().__init__()

        self.encoder = None

        self.Q1 = QFunction(state_dim, action_dim, hidden_dim)
        self.Q2 = QFunction(state_dim, action_dim, hidden_dim)

        self.apply(weight_init)

    def forward(self, observation, action, detach_encoder=False):
        if self.encoder is not None:
            observation = self.encoder(observation, detach=detach_encoder)
        q1 = self.Q1(observation, action)
        q2 = self.Q2(observation, action)

        return q1, q2, observation

    def log(self, L, step, log_freq=LOG_FREQ):
        self.Q1.log(L, step, log_freq)
        self.Q2.log(L, step, log_freq)


class SAC(object):
    def __init__(self, device, state_dim, action_dim, hidden_dim, hidden_depth,
                 initial_temperature, actor_lr, critic_lr, actor_beta,
                 critic_beta, log_std_min, log_std_max, period=1):
        self.device = device

        # self.period = period
        # self.internal = 0

        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
            hidden_depth,
            log_std_min,
            log_std_max).to(device)

        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_dim).to(device)

        self.obs_decoder = None

        self.critic_target = Critic(
            state_dim,
            action_dim,
            hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        # self.internal = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, observation):
        # if self.internal % self.period == 0:
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            observation = observation.unsqueeze(0)
            mu, _, _, _ = self.actor(
                observation, compute_pi=False, compute_log_pi=False)
            action_sel = mu.cpu().data.numpy().flatten()
            # self.last_action_sel = action_sel
            # self.internal += 1
            return action_sel
        # else:
        #     self.internal += 1
        #     return self.last_action_sel

    def sample_action(self, observation):
        # if self.internal % self.period == 0:
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            observation = observation.unsqueeze(0)
            mu, pi, _, _ = self.actor(observation, compute_log_pi=False)
            action_sam = pi.cpu().data.numpy().flatten()
            # self.last_action_sam = action_sam
            # self.internal += 1
            return action_sam
        # else:
        #     self.internal += 1
        #     return self.last_action_sam

    def sample_action_batch(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            observation = observation.unsqueeze(0)
            mu, pi, _, _ = self.actor(observation, compute_log_pi=False)
            return pi.cpu().data.numpy()

    def _update_critic(self, obs, action, reward, next_obs, not_done, discount,
                       L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2, _ = self.critic_target(
                next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * discount * target_V)

        # Get current Q estimates
        current_Q1, current_Q2, h_obs = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # L.info(f"train_critic/loss: {critic_loss}")

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step)
        if self.critic.encoder is not None:
            self.critic.encoder.log(L, step)

    def _update_actor(self, obs, target_entropy, L, step):
        _, pi, log_pi, entropy = self.actor(obs, detach_encoder=True)

        actor_Q1, actor_Q2, _ = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        # L.info(f"train_actor/loss: {actor_loss}")
        # L.info(f"train_actor/entropy {entropy.mean()}")
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        if target_entropy is not None:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                    self.alpha * (-log_pi - target_entropy).detach()).mean()
            # L.info(f"train_alpha/target_entropy: {target_entropy}")
            # L.info(f"train_alpha/loss: {alpha_loss}")
            # L.info(f"train_alpha/value: {self.alpha}")
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self,
               replay_buffer,
               step,
               L,
               batch_size=100,
               discount=0.99,
               tau=0.005,
               policy_freq=2,
               target_entropy=None):

        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            batch_size)

        # L.info(f"train/batch_reward: {reward.mean()}")

        self._update_critic(obs, action, reward, next_obs, not_done, discount,
                            L, step)

        if step % policy_freq == 0:
            self._update_actor(obs, target_entropy, L, step)
            soft_update_params(self.critic, self.critic_target, tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(),
                   "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   "%s/critic_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load("%s/critic_%s.pt" % (model_dir, step)))


def sac_experiment(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    real_env = gym.make(cfg.env.params.name)
    set_seed_everywhere(cfg.random_seed)

    obs_dim = cfg.model.params.dx
    action_dim = cfg.model.params.du
    target_entropy_coef = 1
    batch_size = cfg.alg.params.batch_size  # 512
    discount = cfg.alg.trainer.discount  # .99
    tau = cfg.alg.trainer.tau  # .005
    policy_freq = cfg.alg.trainer.target_update_period  # 2
    replay_buffer_size = int(cfg.alg.replay_buffer_size)  # 1000000
    start_steps = cfg.alg.params.start_steps  # 10000
    eval_freq = cfg.alg.params.eval_freq  # 10000
    max_steps = int(cfg.alg.params.max_steps)  # 2E6
    num_eval_episodes = cfg.alg.params.num_eval_episodes  # 5
    num_eval_timesteps = cfg.alg.params.num_eval_timesteps  # 1000
    num_rl_updates = 1
    model_dir = None

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
                 log_std_max=cfg.alg.trainer.log_std_max,
                 period=cfg.policy.params.period)  # 2)

    step = 0
    steps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_success = 0
    episode_step = 0
    saved_idx = 0
    done = True
    returns = None
    target_entropy = -action_dim * target_entropy_coef

    if cfg.metric.name == 'Living':
        metric = living_reward
    elif cfg.metric.name == 'Rotation':
        metric = rotation_mat
    elif cfg.metric.name == 'Square':
        metric = squ_cost
    else:
        raise ValueError("Improper metric name passed")

    to_plot_rewards = []
    total_steps = []
    rewards = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps, None, metric=metric)

    to_plot_rewards.append(rewards)
    total_steps.append(0)

    env = gym.make(cfg.env.params.name)

    # from gym import spaces
    # env.action_space = spaces.Box(low=np.array([0, 0, 0, 0]),
    #                                    high=np.array([65535, 65535, 65535, 65535]),
    #                                    dtype=np.int32)

    layout = dict(
        title=f"Learning Curve Reward vs Number of Steps Trials (Env: {cfg.env.params.name}, Alg: {cfg.policy.mode})",
        xaxis={'title': f"Steps*{eval_freq}"},
        yaxis={'title': f"Avg Reward Num:{num_eval_episodes}"},
        font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
        legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'}
    )

    while step < max_steps:
        # log.info(f"===================================")
        if step % 1000 == 0:
            log.info(f"Step {step}")

        if done:
            # Evaluate episode
            if steps_since_eval >= eval_freq:
                steps_since_eval %= eval_freq
                log.info(f"eval/episode: {episode_num}")
                returns = evaluate_policy(env, policy, step, log, num_eval_episodes, num_eval_timesteps,
                                          None, metric=metric)
                to_plot_rewards.append(returns)
                total_steps.append(step)

                if model_dir is not None:
                    policy.save(model_dir, step)

            # log.info(f"train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            episode_num += 1

            # log.info(f"train/episode', episode_num, step)

        # Select action randomly or according to policy
        if step < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                with eval_mode(policy):
                    action = policy.sample_action(obs)

        if step >= start_steps:
            num_updates = start_steps if step == start_steps else num_rl_updates
            for _ in range(num_updates):
                policy.update(
                    replay_buffer,
                    step,
                    log,
                    batch_size,
                    discount,
                    tau,
                    policy_freq,
                    target_entropy=target_entropy)

        action_scale = env.action_space.high * (action + 1) / 2
        next_obs, reward, done, _ = env.step(action_scale)
        # print(next_obs[:3])
        # done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        done = 1 if episode_step + 1 == num_eval_timesteps else float(done)
        reward = metric(next_obs, action)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs

        episode_step += 1
        step += 1
        steps_since_eval += 1
        if (step % eval_freq) == 0:
            trial_log = dict(
                env_name=cfg.env.params.name,
                trial_num=saved_idx,
                replay_buffer=replay_buffer if cfg.save_replay else [],
                steps=total_steps,
                policy=policy,
                rewards=to_plot_rewards,
            )
            save_log(cfg, step, trial_log)
            saved_idx += 1

    plot_rewards_over_trials(to_plot_rewards, cfg.env.params.name, save=True)

def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


log = logging.getLogger(__name__)


@hydra.main(config_path='conf/sac.yaml')
def experiment(cfg):
    sac_experiment(cfg)


if __name__ == '__main__':
    sys.exit(experiment())
