#!/usr/bin/env python
"""
POLICY ITERATION
Implement policy evaluation and improvement to find optimal policy.
"""
import gym
import numpy as np

env = gym.make("CartPole-v0")
actions = [0, 1]
alpha = 0.1
gamma = 1.0
epsilon = 0.1
num_iterations = 10
num_episodes = 1000

# initialize policy and value function
policy = {}
value_function = {}

for state in all_possible_states:
    policy[state] = np.random.choice(actions)
    value_function[state] = 0.0

for i in range(num_iterations):
    # Policy evaluation
    for episode in range(num_episodes):
        observation = env.reset()
        state = discretize_state(observation)

