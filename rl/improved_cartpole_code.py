#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# CART POLE
env = gym.make('CartPole-v1')
print(env.observation_space.low,"\n",env.observation_space.high)

def Qtable(state_space, action_space, bin_size=30):
    # Define bins with normalization for the extreme velocity values
    bins = [
        np.linspace(-4.8, 4.8, bin_size),              # Cart Position
        np.linspace(-3.0, 3.0, bin_size),              # Cart Velocity (scaled)
        np.linspace(-0.418, 0.418, bin_size),          # Pole Angle
        np.linspace(-3.0, 3.0, bin_size)               # Pole Velocity at Tip (scaled)
    ]
    q_table = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    return q_table, bins

def Discrete(state, bins):
    index = []
    for i in range(len(state)): 
        state_value = np.clip(state[i], bins[i][0], bins[i][-1])  # Clip values to ensure within bounds
        index.append(np.digitize(state_value, bins[i]) - 1)  # Discretize state
    return tuple(index)

# TRAINING
q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n)

def Q_learning(q_table, bins, episodes=5000, gamma=0.95, lr=0.1, timestep=5000, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    rewards = 0
    solved = False
    steps = 0
    runs = [0]
    data = {'max': [0], 'avg': [0]}
    start = time.time()
    ep = [i for i in range(0, episodes + 1, timestep)]
    for episode in range(1, episodes+1):
        current_state, _ = env.reset()  # Get only the observation
        current_state = Discrete(current_state, bins)  # Discretize the initial observation
        score = 0
        done = False
        temp_start = time.time()
        while not done:
            steps += 1
            ep_start = time.time()
            
            # Action selection with epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                action = np.argmax(q_table[current_state])  # Exploitation

            next_state, reward, done, _, _ = env.step(action)
            next_state = Discrete(next_state, bins)
            score += reward
            
            # Q-Learning Update Rule
            if done and score < 500:
                q_table[current_state][action] += lr * (reward - q_table[current_state][action])
            else:
                q_table[current_state][action] += lr * (reward + gamma * np.max(q_table[next_state]) - q_table[current_state][action])
            
            current_state = next_state
            
            if done:
                rewards += score
                if episode % timestep == 0:
                    avg_reward = rewards / timestep
                    max_reward = np.max(data['max'])
                    print(f"Episode : {episode} | Reward -> {avg_reward} | Max reward : {max_reward} | Time : {time.time() - temp_start}")
                    rewards = 0

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Track max reward achieved
        data['max'].append(score)
        data['avg'].append(score)
    
    end = time.time()
    print(f"Solved in episode : {episode} in time {end - start}")

Q_learning(q_table, bins)
