import torch
import torch.nn as nn
import gym
import numpy as np

from torch.optim import Adam
from stone.utils.logx import EpochLogger

from stone.cores import ac_core
from copy import deepcopy
from itertools import chain
import time


class td3:
    def __init__(self, env_fn, hid=256, layers=2, gamma=0.99, seed=0, buffer_size=int(1e6), polyak=0.995,
                 batch_size=100, logger_kwargs=dict(), activation=nn.ReLU, update_every=50, pi_lr=1e-3, q_lr=1e-3,
                 update_after=1000, noise_scale=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
                 start_steps=10000):
        super(td3, self).__init__()

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # set up environment
        self.env, self.test_env = env_fn(), env_fn()  # must separate env for training and env for testing
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed), self.test_env.seed(seed)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # create actor-critic
        self.ac = ac_core.MLPDoubleActorCritic(self.obs_dim[0], self.act_dim[0], [hid] * layers, activation,
                                               self.act_limit)

        # create target-Q and target-actor
        self.ac_target = deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        # Count variables
        var_counts = tuple(ac_core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # set up replay buffer
        self.buf = ac_core.ac_buffer(buffer_size, self.obs_dim, self.act_dim)

        # optimizers
        self.q_params = chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # training configurations
        self.gamma = gamma
        self.polyak = polyak
        self.noise_scale = noise_scale
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.start_steps = start_steps

    def make_action(self, o, deterministic=False):
        a = self.ac.act(o)
        if not deterministic:
            a += self.noise_scale * np.random.randn(self.act_dim[0])
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_policy(self, max_ep_len, num_test_episodes):
        for e in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                a = self.make_action(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train(self, epochs=500, steps_per_epoch=4000, max_ep_len=1000, num_test_episodes=10, save_freq=10):

        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for t in range(epochs * steps_per_epoch):
            if t > self.start_steps:
                a = self.make_action(torch.as_tensor(o, dtype=torch.float32))
            else:
                a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # This is very important! because HalfCheetah will return done after 1000 steps whatever it is really done!
            d = False if ep_len == max_ep_len else d

            # save and log
            self.buf.store(o, a, r, o2, d)

            # update obs
            o = o2

            if t % self.update_every == 0 and t > self.update_after:
                self.update()

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                self.test_policy(max_ep_len, num_test_episodes)

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs - 1):
                    self.logger.save_state({'env': self.env}, None)

                # Log info about epoch
                logger = self.logger
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('QVals', with_min_and_max=True)
                logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()

    def update(self):

        # Train with a few steps of gradient descent
        for i in range(self.update_every):
            data = self.buf.sample(self.batch_size)
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['next_obs'], data['done']

            # update Q-network
            self.q_optimizer.zero_grad()
            with torch.no_grad():
                act_noise = torch.randn(self.batch_size, self.act_dim[0]) * self.target_noise
                act_noise = torch.clamp(act_noise, -self.noise_clip, self.noise_clip)
                a2 = self.ac_target.pi(o2) + act_noise
                a2 = torch.clamp(a2, -self.act_limit, self.act_limit)
                q1_target = self.ac_target.q1(o2, a2)
                q2_target = self.ac_target.q2(o2, a2)
                q_target = torch.min(q1_target, q2_target)
                backup = r + self.gamma * (1 - d) * q_target
            q1 = self.ac.q1(o, a)
            q2 = self.ac.q2(o, a)
            loss_q1 = ((backup - q1) ** 2).mean()
            loss_q2 = ((backup - q2) ** 2).mean()
            loss_q = loss_q1 + loss_q2
            loss_q.backward()
            self.q_optimizer.step()

            # update policy every policy_delay Q updates
            if i % self.policy_delay == 0:
                # Freeze Q-network so you don't waste computational effort
                # computing gradients for it during the policy learning step.
                for params in self.q_params:
                    params.requires_grad = False

                self.pi_optimizer.zero_grad()
                loss_pi = -self.ac.q1(o, self.ac.pi(o)).mean()
                loss_pi.backward()
                self.pi_optimizer.step()

                # unfreeze params after updating policy
                for params in self.q_params:
                    params.requires_grad = True

                # update target network by polyak averaging
                with torch.no_grad():
                    for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

        self.logger.store(LossPi=loss_pi.item(), LossQ=loss_q.item(), QVals=q1.detach().numpy())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='Ant-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=1e-3)
    parser.add_argument('--q_lr', '-v', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--exp_name', type=str, default='td3_HalfCheetah')
    parser.add_argument('--exp_name', type=str, default='td3_Ant')
    args = parser.parse_args()

    from stone.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = td3(lambda: gym.make(args.env), hid=args.hid, layers=args.layers, seed=args.seed,
                pi_lr=args.pi_lr, q_lr=args.q_lr, logger_kwargs=logger_kwargs)
    agent.train(epochs=args.epochs)
