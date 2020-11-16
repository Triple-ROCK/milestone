import torch
import gym
import numpy as np

from torch.optim import Adam
from stone.single_thread._buffers import GAE_buffer
from stone.utils.logx import EpochLogger

from stone.cores import pg_core
import time


class ppo:
    def __init__(self, env_fn, hid=64, layers=2, gamma=0.99, lam=0.97, seed=0,
                 steps_per_epoch=4000, pi_lr=3e-4, v_lr=1e-3, clip_ratio=0.2, logger_kwargs=dict()):
        super(ppo, self).__init__()

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # set up environment
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        # ppo clip ratio
        self.clip_ratio = clip_ratio

        # create actor-critic
        self.ac = pg_core.MLPActorCritic(self.env, hidden_sizes=[hid] * layers)

        # Count variables
        var_counts = tuple(pg_core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # set up replay buffer
        self.steps_per_epoch = steps_per_epoch
        self.buf = GAE_buffer(steps_per_epoch, self.obs_dim, self.act_dim, gamma, lam)

        # optimizers
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=v_lr)

    def train(self, epochs=500, train_v_iters=80, train_pi_iters=80, max_ep_len=1000, target_kl=0.01):

        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for epoch in range(epochs):
            for t in range(self.steps_per_epoch):
                o_torch = torch.as_tensor(o, dtype=torch.float32)
                a, v, logp = self.ac.step(o_torch)
                next_o, r, d, _ = self.env.step(a)

                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                self.logger.store(VVals=v)

                # update obs
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.\n' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0.
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            self.update(train_pi_iters, train_v_iters, target_kl)

            # Log info about epoch
            logger = self.logger
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = (logp - logp_old).exp()
        adv_clip = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -torch.min(ratio * adv, adv_clip).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, rtg = data['obs'], data['rtg']
        return ((self.ac.v(obs) - rtg) ** 2).mean()

    def update(self, train_pi_iters, train_v_iters, target_kl):
        data = self.buf.get()

        # Train policy with a few steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
            if pi_info['kl'] > 1.5 * target_kl:
                print('PPO stops at iter %d' % i)
                break

        # Value function learning
        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info['ent']
        self.logger.store(LossPi=loss_pi, LossV=loss_v, KL=kl, Entropy=ent)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=3e-4)
    parser.add_argument('--v_lr', '-v', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='_ppo')
    args = parser.parse_args()

    from stone.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = ppo(lambda: gym.make(args.env), hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr, logger_kwargs=logger_kwargs)
    agent.train(epochs=args.epochs)
