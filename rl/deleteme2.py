#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
np.random.seed(73)

# CartPole environment
env = gym.make("CartPole-v1")
env.reset(seed=73)  # Reproducibility
# Outputs the lower/upper bounds of observation space
print(env.observation_space.low, "\n", env.observation_space.high)  # |S|,|A(s)|

def Qtable(state_space=len(env.observation_space.low), action_space=env.action_space.n, bin_size=30):
    """
    Q-TABLE: Defined by the number of state and actions...
    ------------------------------------------------------
    Each element Q[i,j,k,l,m] represents the Q-value for:
        Cart position in bin i
        Cart velocity in bin j
        Pole angle in bin k
        Pole angular velocity in bin l
        Action m (0 or 1, corresponding to left or right)
    ----------------------------------------------------------------
    BINS: List of numpy arrays representing the discretized boundaries for one
    of the state variables in env., a numpy array for each state.
    ----------------------------------------------------------------
    INPUT:
        state_space: (int); default: 4 -> {cart_pos, cart_vel, pole_pos,
            pole_angle_vel}
        action_space: (int); default: 2 -> {right, left}
        bin_size: (int); default: 30

    OUTPUT:
        q_table, bins: (tuple of ndarrays ?, list of ndarrays)
    """
    # Adjusted ranges for the state variables
    bins = [
        # Cart position (bounded between -4.8 and 4.8)
        np.linspace(-4.8, 4.8, bin_size),
        # Cart velocity (limited manually)
        np.linspace(-3, 3, bin_size),
        # Pole angle (bounded between -0.418 and 0.418 radians)
        np.linspace(-0.418, 0.418, bin_size),
        # Pole angular velocity (limited manually)
        np.linspace(-4, 4, bin_size)
    ]

    # Initialize Q-table with zeros
    q_table = np.zeros([bin_size] * state_space + [action_space])

    return q_table, bins  # q_table.ndim == 5, list of 4 np arrays

def discrete(state, bins):
    """
    Discretizes continuous space
    ----------------------------
    INPUT:
        state: (array) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        indices: (tuple) for making it hashable for indexing into multidimensional
        (5-dimensional table, here: q_table.ndim) Q-table.
    """
    index = []

    for i in range(len(state)):
        # Clip the state values to be within the bins range
        value = max(min(state[i], bins[i][-1]), bins[i][0])
        # Corresponding Indices of bins; discretization process
        index.append(np.digitize(value, bins[i]) - 1)

    indices = tuple(index)

    return indices

def q_learning(q_table, bins, episodes=5000, discount=0.9, alpha=0.1,
               timestep=100, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """
    Learns iteratively, the optimal Q-value function using the Bellman
    equation via storing Q-values in the Q-table to be updated @ each time
    step.
    ------------------------------------------------------------
    INPUT:
        q_table: (np.arrays) initialized Q-table
        bins: (list of np.arrays) Discretized bins for state variables
        episodes: (int) Number of episodes to train over
        discount: (float) Discount factor determining importance of future
            rewards
        alpha: (float) Learning rate
        timestep: (int) Interval reporting training progress
        epsilon: Initial exploration rate
        epsilon_decay: (float) Rate at which epsilon decays
        epsilon_min: (float) Minimum value of epsilon

    OUTPUT:
        episode_rewards: (list) Total reward per episode
        q_table: (np.array) Updated Q-table
    """
    episode_rewards = []

    for episode in range(1, episodes + 1):
        # Initialize indices for Q-table
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the action with max Q-value
                action = np.argmax(q_table[current_state])

            # Execute chosen action
            next_state_continuous, reward, done, _, _ = env.step(action)
            # Discretize new continuous state for indexing Q-table
            next_state = discrete(next_state_continuous, bins)
            # Find action with highest Q-value in next state
            best_next_action = np.argmax(q_table[next_state])
            # TD target
            td_target = reward + discount * q_table[next_state + (best_next_action,)]
            # TD error
            td_error = td_target - q_table[current_state + (action,)]
            # Update Q-value
            q_table[current_state + (action,)] += alpha * td_error

            # Update current state and total reward
            current_state = next_state
            total_reward += reward

        # Decay epsilon to reduce exploration over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min

        episode_rewards.append(total_reward)

        # Output progress at every timestep
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()

    return episode_rewards, q_table

def sarsa(q_table, bins, episodes=5000, discount=0.9, alpha=0.1,
          timestep=100, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """
    Implements the SARSA algorithm.
    """
    episode_rewards = []

    for episode in range(1, episodes + 1):
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)

        # Choose action using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        total_reward = 0
        done = False

        while not done:
            next_state_continuous, reward, done, _, _ = env.step(action)
            next_state = discrete(next_state_continuous, bins)

            # Choose next action using epsilon-greedy policy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            # Update Q-value
            td_target = reward + discount * q_table[next_state + (next_action,)]
            td_error = td_target - q_table[current_state + (action,)]
            q_table[current_state + (action,)] += alpha * td_error

            current_state = next_state
            action = next_action
            total_reward += reward

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min

        episode_rewards.append(total_reward)

        # Output progress at every timestep
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    return episode_rewards, q_table

def moving_average(data, window_size):
    """
    Computes the moving average of the data using the specified window size.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    q_table_q_learning, bins = Qtable()
    q_table_sarsa, _ = Qtable()
    breakpoint()

    episodes = 5000

    # Q-Learning
    rewards_q_learning, trained_q_table_q = q_learning(
        q_table_q_learning, bins, episodes=episodes)

    # SARSA
    rewards_sarsa, trained_q_table_sarsa = sarsa(
        q_table_sarsa, bins, episodes=episodes)

    # Plotting the rewards with moving average
#    window_size = 37
#
#    plt.figure(figsize=(12, 8))
#    plt.plot(moving_average(rewards_q_learning, window_size), label='Q-Learning')
#    plt.plot(moving_average(rewards_sarsa, window_size), label='SARSA')
#    plt.xlabel('Episode')
#    plt.ylabel('Total Reward (Moving Average)')
#    plt.title('Performance Comparison of Tabular Methods')
#    plt.legend()
#    plt.show()

if __name__ == "__main__":
    main()


