#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize the CartPole environment
env = gym.make('CartPole-v1')
print(env.observation_space.low, "\n", env.observation_space.high)

def Qtable(state_space, action_space, bin_size=30):
    """
    Initializes the Q-table with random values and defines bins for discretizing continuous state variables.

    Parameters:
    - state_space: Number of state variables (dimensions of the observation space).
    - action_space: Number of possible actions.
    - bin_size: Number of discrete bins for each state variable.

    Returns:
    - q_table: The initialized Q-table.
    - bins: A list of arrays representing the bin edges for each state variable.
    """
    # Define bins for each continuous state variable
    bins = [
        np.linspace(-4.8, 4.8, bin_size),      # Cart Position
        np.linspace(-3.0, 3.0, bin_size),      # Cart Velocity
        np.linspace(-0.418, 0.418, bin_size),  # Pole Angle (approx -24 to 24 degrees in radians)
        np.linspace(-3.0, 3.0, bin_size)       # Pole Velocity at Tip
    ]
    # Initialize Q-table with random values
    q_table = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    return q_table, bins

def Discrete(state, bins):
    """
    Converts continuous state variables into discrete indices using the provided bins.

    Parameters:
    - state: The continuous state observed from the environment.
    - bins: The discretization bins for each state variable.

    Returns:
    - A tuple of indices representing the discretized state.
    """
    index = []
    for i in range(len(state)):
        # Clip state values to be within the bins' range
        state_value = np.clip(state[i], bins[i][0], bins[i][-1])
        # Discretize the state variable
        index.append(np.digitize(state_value, bins[i]) - 1)
    return tuple(index)

# Initialize Q-table and bins
q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n)

def Q_learning(q_table, bins, episodes=5000, gamma=0.95, lr=0.2, timestep=5000, epsilon=0.8, epsilon_decay=0.99, epsilon_min=0.01):
    """
    Performs the Q-Learning algorithm to train the agent on the CartPole environment.

    Parameters:
    - q_table: The initialized Q-table.
    - bins: The discretization bins for state variables.
    - episodes: Number of episodes to train over.
    - gamma: Discount factor for future rewards.
    - lr: Learning rate.
    - timestep: Interval for reporting training progress.
    - epsilon: Initial exploration rate.
    - epsilon_decay: Rate at which epsilon decays after each episode.
    - epsilon_min: Minimum value of epsilon.

    Returns:
    - None
    """
    rewards = 0
    steps = 0
    start = time.time()

    for episode in range(1, episodes + 1):
        current_state, _ = env.reset()
        current_state = Discrete(current_state, bins)
        score = 0
        done = False

        while not done:
            steps += 1
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore: select a random action
            else:
                action = np.argmax(q_table[current_state])  # Exploit: select the action with max Q-value

            # Take action and observe the outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = Discrete(next_state, bins)
            score += reward

            # Update Q-value using the Q-Learning update rule
            if done and score < 500:
                q_table[current_state][action] += lr * (reward - q_table[current_state][action])
            else:
                q_table[current_state][action] += lr * (reward + gamma * np.max(q_table[next_state]) - q_table[current_state][action])

            current_state = next_state

        rewards += score

        # Decay epsilon to reduce exploration over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Reporting progress at specified intervals
        if episode % timestep == 0:
            avg_reward = rewards / timestep
            print(f"Episode: {episode}, Average Reward: {avg_reward}, Time Elapsed: {time.time() - start}")
            rewards = 0

# Run the Q-Learning algorithm
Q_learning(q_table, bins)


