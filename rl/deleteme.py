#!/usr/bin/env python
import matplotlib.pyplot as plt

# Collect rewards during training
rewards_sarsa = []
rewards_q_learning = []

# Example for SARSA
for episode in range(num_episodes):
    total_reward = 0
    # SARSA training loop...
    rewards_sarsa.append(total_reward)

# Compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

ma_rewards_sarsa = moving_average(rewards_sarsa, window=100)
ma_rewards_q_learning = moving_average(rewards_q_learning, window=100)

# Plot learning curves
plt.plot(ma_rewards_sarsa, label='SARSA')
plt.plot(ma_rewards_q_learning, label='Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Average Return')
plt.title('Learning Curves')
plt.legend()
plt.show()


