#!/usr/bin/env python
"""
POLICY ITERATION Assignement:
Implement policy evaluation and improvement to find optimal policy.
----------------------------------------------------------------------
Learning Objectives
• Implement and understand policy iteration, temporal difference (TD) learning, and Monte Carlo (MC)
methods.
• Apply discretization techniques to handle continuous state spaces in tabular methods.
• Utilize function approximation to manage high-dimensional or continuous state spaces.
• Compare the performance of different RL algorithms and understand their advantages and limitations.
"""
import gym
import numpy as np

env = gym.make("CartPole-v0")
# My attempt ?
state_space = 4
action_space = 2

def Qtable(state_space, action_space, bin_size=30):
    bins = [
        np.linspace(-4.8, 4.8, bin_size),
        np.linspace(-4, 4, bin_size),
        np.linspace(-0.418, 0.418, bin_size),
        np.linspace(-4, 4, bin_size)
    ]

    q_table = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space
                                                     + [action_space]))

    return q_table, bins

def discrete(state, bins):
    index = []

    for i in range(len(state)):
        index.append(np.digitize(state[i], bins[i]) - 1)

    return tuple(index)


