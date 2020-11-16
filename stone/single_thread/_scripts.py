import gym
import argparse

from stone.single_thread._vpg import vpg


def REINFORCE():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=1e-2)
    parser.add_argument('--v_lr', '-v', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = vpg(env, hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr)
    agent.train(epochs=args.epochs)


def A2C():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', '-l', type=int, default=2)
    parser.add_argument('--pi_lr', '-pi', type=float, default=1e-2)
    parser.add_argument('--v_lr', '-v', type=float, default=1e-2)
    parser.add_argument('--lam', type=float, default=0.)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = vpg(env, hid=args.hid, layers=args.layers, seed=args.seed, lam=args.lam,
                steps_per_epoch=args.steps, pi_lr=args.pi_lr, v_lr=args.v_lr)
    agent.train(epochs=args.epochs)


if __name__ == "__main__":
    # REINFORCE()
    A2C()
