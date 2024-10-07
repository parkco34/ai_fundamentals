#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# Suppress Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# CART POLE
env = gym.make('CartPole-v1')

def Qtable(state_space, action_space, bin_size=30):
    # Define reasonable bounds for velocities
    cart_pos_bins = np.linspace(-4.8, 4.8, bin_size)
    cart_vel_bins = np.linspace(-3.0, 3.0, bin_size)
    pole_angle_bins = np.linspace(-0.418, 0.418, bin_size)
    pole_vel_bins = np.linspace(-4.0, 4.0, bin_size)
    bins = [cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins]
    return bins

def Discrete(state, bins):
    index = []
    for i in range(len(state)):
        state_value = np.clip(state[i], bins[i][0], bins[i][-1])
        index.append(np.digitize(state_value, bins[i]) - 1)
    return tuple(index)

bins = Qtable(len(env.observation_space.low), env.action_space.n)

def Q_learning(bins, episodes=5000, gamma=0.99, lr=0.1, timestep=500, epsilon=1.0):
    q_table = {}
    rewards_list = []
    solved = False
    start_time = time.time()
    # for slower epsilon decay
    epsilon_decay = 0.999
    for episode in range(1, episodes+1):
        current_state, _ = env.reset()
        current_state = Discrete(current_state, bins)
        score = 0
        done = False
        while not done:
            # Initialize Q-values for unseen states
            if current_state not in q_table:
                q_table[current_state] = np.zeros(env.action_space.n)
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])
            # Take action
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = Discrete(observation, bins)
            score += reward
            # Initialize Q-values for the next state if unseen
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)
            # Q-Learning update
            max_future_q = np.max(q_table[next_state])
            current_q = q_table[current_state][action]
            new_q = (1 - lr) * current_q + lr * (reward + gamma * max_future_q * (not done))
            q_table[current_state][action] = new_q
            current_state = next_state
        # End of the episode
        rewards_list.append(score)
        epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon
        # Check if solved
        if episode >= 100:
            average_reward = np.mean(rewards_list[-100:])
            if average_reward >= 475 and not solved:
                solved = True
                print(f'Solved in episode: {episode}')
        # Logging progress
        if episode % timestep == 0:
            average_reward = np.mean(rewards_list[-timestep:])
            print(f'Episode: {episode}, Average Reward: {average_reward:.2f}, Epsilon: {epsilon:.4f}')
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    # Plotting the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.show()
    env.close()

Q_learning(bins, lr=0.05, gamma=0.99, episodes=10000, timestep=500)

#breakpoint()
