import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.functional import softplus


class ac_buffer:
    def __init__(self, size, obs_dim, act_dim):
        """
        A buffer for storing trajectories experienced by a actor-critic agent interacting
        with the environment
        :param size: number of datapoints
        :param obs_dim: scalar / tuple
        :param act_dim: scalar / tuple
        """
        self.obs_buf = np.zeros(combined_shape(size, obs_dim))
        self.act_buf = np.zeros(combined_shape(size, act_dim))
        self.rew_buf = np.zeros((size,))
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim))
        self.done_buf = np.zeros((size,))

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        store one transition into buffer
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch_idx = np.random.randint(0, self.size, batch_size)
        obs, act, rew, next_obs, done = self.obs_buf[batch_idx], self.act_buf[batch_idx], \
                                        self.rew_buf[batch_idx], self.next_obs_buf[batch_idx], self.done_buf[batch_idx]
        data = dict(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


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


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.pi(obs) * self.act_limit


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class MLPDoubleActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class MLPSacActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
        self.pi = mlp(pi_sizes, activation)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # Return output from network scaled to action space limits.
        out = self.pi(obs)
        mu, log_std = torch.split(out, out.shape[-1] // 2, dim=-1)  # You can't fit std directly, use log std!
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        pi = Normal(mu, torch.exp(log_std))
        act = mu if deterministic else pi.rsample()

        if with_logprob:
            logp = (pi.log_prob(act) - 2 * (np.log(2) - act - softplus(-2 * act))
                    ).sum(dim=-1)
        else:
            logp = None
        return torch.tanh(act) * self.act_limit, logp


class MLPSacActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = MLPSacActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
