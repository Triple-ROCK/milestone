import numpy as np

import torch
import torch.nn as nn

from gym.spaces import Discrete, Box
from stone.utils.mpi_tools import mpi_statistics_scalar
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class GAE_buffer:
    def __init__(self, size, obs_dim, act_dim, gamma, lam):
        """
        A buffer for storing trajectories experienced by a pg agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
        for calculating the advantages of state-action pairs.
        :param size: number of datapoints
        :param obs_dim: scalar
        :param act_dim: scalar
        :param gamma: discount factor
        :param lam: GAE-lambda, lam=1 means REINFORCE and lam=0 means A2C, typically 0.9~0.99
        """
        self.obs_buf = np.zeros(combined_shape(size, obs_dim))
        self.act_buf = np.zeros(combined_shape(size, act_dim))
        self.rew_buf = np.zeros((size,))
        self.val_buf = np.zeros((size,))
        self.rtg_buf = np.zeros((size,))
        self.adv_buf = np.zeros((size,))
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        store one transition into buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, val):
        """
        Compute reward-to-go whenever the following cases appear
         1. agent dies, which means the return following is zero.
         2. it reaches the max_episode_len or the trajectory being cut off at time T,
            then you should provided an estimate V(S_T) using critic to compensate for the rewards beyond time T
        :param v: the value estimated by critic for the final state
        :return: None
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], val)
        vals = np.append(self.val_buf[path_slice], val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.rtg_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, 'You must fulfill buffer before getting data!'
        self.path_start_idx, self.ptr = 0, 0

        adv_mu, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mu) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def discount_cumsum(xs, gamma):
    ys = np.zeros_like(xs, dtype=np.float32)
    cumsum = 0.
    for i, x in enumerate(xs[::-1]):
        cumsum *= gamma
        cumsum += x
        ys[-1 - i] = cumsum
    return ys


def statistics_scalar(X, with_min_and_max=False):
    mu = np.mean(X)
    std = np.std(X)
    if with_min_and_max:
        min = np.min(X)
        max = np.max(X)
        return mu, std, min, max
    return mu, std


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, env, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = env.observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(env.action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, env.action_space.shape[0], hidden_sizes, activation)
        elif isinstance(env.action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, env.action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
