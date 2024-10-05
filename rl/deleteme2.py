#!/usr/bin/env python
import gym
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Specify the exact version of the environment
env = gym.make('CartPole-v1')
actions = [0, 1]
alpha = 0.1
gamma = 1.0
epsilon = 0.1
num_iterations = 10
num_episodes = 1000
num_bins = 10

# Define bins for each state variable
cart_pos_bins = np.linspace(-4.8, 4.8, num_bins - 1)
cart_vel_bins = np.linspace(-3.0, 3.0, num_bins - 1)
pole_angle_bins = np.linspace(-0.418, 0.418, num_bins - 1)
pole_vel_bins = np.linspace(-3.0, 3.0, num_bins - 1)

# Append -inf and inf to include all possible values
cart_pos_bins = np.concatenate(([-np.inf], cart_pos_bins, [np.inf]))
cart_vel_bins = np.concatenate(([-np.inf], cart_vel_bins, [np.inf]))
pole_angle_bins = np.concatenate(([-np.inf], pole_angle_bins, [np.inf]))
pole_vel_bins = np.concatenate(([-np.inf], pole_vel_bins, [np.inf]))

def discretize_state(observation):
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    # Discretize each variable
    cart_pos_disc = np.digitize(cart_pos, cart_pos_bins) - 1
    cart_vel_disc = np.digitize(cart_vel, cart_vel_bins) - 1
    pole_angle_disc = np.digitize(pole_angle, pole_angle_bins) - 1
    pole_vel_disc = np.digitize(pole_vel, pole_vel_bins) - 1

    # Combine discretized variables into a single state tuple
    state = (cart_pos_disc, cart_vel_disc, pole_angle_disc, pole_vel_disc)
    return state

# Generate all possible discrete states
state_bins = [range(num_bins) for _ in range(4)]  # 4 state variables
all_possible_states = list(product(*state_bins))

# Initialize policy and value function
policy = {}
value_function = {}
for state in all_possible_states:
    policy[state] = np.random.choice(actions)
    value_function[state] = 0.0

# Define epsilon-greedy action selection
def choose_action(Q, state, epsilon):
    if state not in Q:
        Q[state] = np.zeros(len(actions))
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])

# SARSA Algorithm
Q_sarsa = {}
rewards_sarsa = []
for episode in range(num_episodes):
    observation = env.reset()
    if isinstance(observation, tuple):  # Handle potential changes in gym API
        observation = observation[0]
    state = discretize_state(observation)
    action = choose_action(Q_sarsa, state, epsilon)
    total_reward = 0
    done = False
    while not done:
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_observation)
        next_action = choose_action(Q_sarsa, next_state, epsilon)
        if state not in Q_sarsa:
            Q_sarsa[state] = np.zeros(len(actions))
        if next_state not in Q_sarsa:
            Q_sarsa[next_state] = np.zeros(len(actions))
        Q_sarsa[state][action] += alpha * (
            reward + gamma * Q_sarsa[next_state][next_action] * (not done) - Q_sarsa[state][action]
        )
        state = next_state
        action = next_action
        total_reward += reward
    rewards_sarsa.append(total_reward)

# Q-Learning Algorithm
Q_q_learning = {}
rewards_q_learning = []
for episode in range(num_episodes):
    observation = env.reset()
    if isinstance(observation, tuple):  # Handle potential changes in gym API
        observation = observation[0]
    state = discretize_state(observation)
    total_reward = 0
    done = False
    while not done:
        action = choose_action(Q_q_learning, state, epsilon)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_observation)
        if state not in Q_q_learning:
            Q_q_learning[state] = np.zeros(len(actions))
        if next_state not in Q_q_learning:
            Q_q_learning[next_state] = np.zeros(len(actions))
        best_next_action = np.argmax(Q_q_learning[next_state])
        Q_q_learning[state][action] += alpha * (
            reward + gamma * Q_q_learning[next_state][best_next_action] * (not done) - Q_q_learning[state][action]
        )
        state = next_state
        total_reward += reward
    rewards_q_learning.append(total_reward)

# Plotting Learning Curves
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

ma_rewards_sarsa = moving_average(rewards_sarsa, window_size=100)
ma_rewards_q_learning = moving_average(rewards_q_learning, window_size=100)

plt.plot(ma_rewards_sarsa, label='SARSA')
plt.plot(ma_rewards_q_learning, label='Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Average Return')
plt.title('Learning Curves')
plt.legend()
plt.show()

env.close()
