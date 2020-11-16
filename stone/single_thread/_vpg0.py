import torch
import gym
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
from stone.cores import pg_core


class GAE_buffer():
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
        self.obs_buf = np.zeros(pg_core.combined_shape(size, obs_dim))
        self.act_buf = np.zeros(pg_core.combined_shape(size, act_dim))
        self.rew_buf = np.zeros((size,))
        self.val_buf = np.zeros((size,))
        self.rtg_buf = np.zeros((size,))
        self.adv_buf = np.zeros((size,))

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        """
        store one transition into buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
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

        self.adv_buf[path_slice] = pg_core.discount_cumsum(deltas, self.gamma * self.lam)
        self.rtg_buf[path_slice] = pg_core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, 'You must fulfill buffer before getting data!'
        self.path_start_idx, self.ptr = 0, 0

        adv_mu, adv_std = pg_core.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mu) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf,
                    adv=self.adv_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class vpg:
    def __init__(self, env_fn, hid=256, layers=2, gamma=0.99, lam=0.97,
                 seed=0, steps_per_epoch=4000, pi_lr=1e-2, v_lr=1e-3):
        super(vpg, self).__init__()

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = env_fn()
        self.env.seed(seed)
        self.discrete = isinstance(self.env.action_space, Discrete)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        # create actor-critic
        self.mlp_sizes = [self.obs_dim] + [hid] * layers
        self.log_std = torch.nn.Parameter(-0.5 * torch.ones(self.act_dim, dtype=torch.float32))
        self.pi = pg_core.mlp(sizes=self.mlp_sizes + [self.act_dim], activation=nn.Tanh)
        self.v = pg_core.mlp(sizes=self.mlp_sizes + [1], activation=nn.Tanh)

        # Count variables
        var_counts = tuple(pg_core.count_vars(module) for module in [self.pi, self.v])
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Discrete action in buf is of shape (N, )
        self.steps_per_epoch = steps_per_epoch
        self.buf = GAE_buffer(steps_per_epoch, self.obs_dim, self.env.action_space.shape, gamma, lam)

        # optimizers
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.v.parameters(), lr=v_lr)

    def act(self, obs):
        """Used for collecting trajectories or testing, which doesn't require tracking grads"""
        with torch.no_grad():
            logits = self.pi(obs)
            if self.discrete:
                pi = Categorical(logits=logits)
            else:
                pi = Normal(loc=logits, scale=self.log_std.exp())
            act = pi.sample()
        return act.numpy()

    def log_prob(self, obs, act):
        """
        Compute log prob for the given batch observations and actions.
        :param obs: Assume shape (N, D_0)
        :param act: Assume shape (N, D_a)
        :return: log_prob of shape (N,)
        """
        act = act.squeeze(dim=-1)  # critical for discrete actions!
        logits = self.pi(obs)
        if self.discrete:
            pi = Categorical(logits=logits)
            return pi.log_prob(act)
        else:
            pi = Normal(loc=logits, scale=self.log_std.exp())
            return pi.log_prob(act).sum(dim=-1)

    def train(self, epochs=50, train_v_iters=80, max_ep_len=1000):

        ret_stat, len_stat = [], []
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for e in range(epochs):
            for t in range(self.steps_per_epoch):
                o_torch = torch.as_tensor(o, dtype=torch.float32)
                a = self.act(o_torch)
                v = self.v(o_torch).detach().numpy()
                next_o, r, d, _ = self.env.step(a)

                ep_ret += r
                ep_len += 1
                self.buf.store(o, a, r, v)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.\n' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        v = self.v(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
                    else:
                        v = 0.
                    if terminal:
                        ret_stat.append(ep_ret)
                        len_stat.append(ep_len)
                    self.buf.finish_path(v)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            data = self.buf.get()
            loss_pi = self.update_pi(data)
            loss_v = self.update_v(data, train_v_iters)
            print('epoch: %3d \t loss of pi: %.3f \t loss of v: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (e, loss_pi, loss_v, np.mean(ret_stat), np.mean(len_stat)))
            ret_stat, len_stat = [], []

    def update_pi(self, data):
        obs, act, adv = data['obs'], data['act'], data['adv']
        self.pi_optimizer.zero_grad()
        logp = self.log_prob(obs, act)
        loss_pi = (-logp * adv).mean()
        loss_pi.backward()
        self.pi_optimizer.step()
        return loss_pi.item()

    def update_v(self, data, iter):
        obs, rtg = data['obs'], data['rtg']
        for i in range(iter):
            self.v_optimizer.zero_grad()
            v = self.v(obs).squeeze()
            loss_v = ((v - rtg) ** 2).mean()
            loss_v.backward()
            self.v_optimizer.step()
        return loss_v.item()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=1e-2)
    parser.add_argument('--v_lr', '-v', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    agent = vpg(lambda: gym.make(args.env), hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr)
    agent.train(epochs=args.epochs)
