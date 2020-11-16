import torch
from stone.cores import pg_core
import numpy as np


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
        self.obs_buf = np.zeros(pg_core.combined_shape(size, obs_dim))
        self.act_buf = np.zeros(pg_core.combined_shape(size, act_dim))
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

        self.adv_buf[path_slice] = pg_core.discount_cumsum(deltas, self.gamma * self.lam)
        self.rtg_buf[path_slice] = pg_core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, 'You must fulfill buffer before getting data!'
        self.path_start_idx, self.ptr = 0, 0

        adv_mu, adv_std = pg_core.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mu) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}