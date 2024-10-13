#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Neural Network model for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=24):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # No activation on the output layer (linear output)

# Function for TD Learning with a Neural Network
def td_neural_network(env, episodes=1000, gamma=0.99, alpha=0.001):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_size = 24

    # Initialize the neural network
    q_network = QNetwork(state_size, action_size, hidden_size)
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action_values = q_network(state)
                action = torch.argmax(action_values).item()

            # Take action and observe result
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            total_reward += reward

            # Compute TD target
            with torch.no_grad():
                next_action_values = q_network(next_state)
                if done:
                    target = torch.tensor([reward])
                else:
                    target = reward + gamma * torch.max(next_action_values)

            # Current Q-value prediction
            q_values = q_network(state)
            q_value = q_values[0, action]

            # Loss and backpropagation
            loss = loss_fn(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move to the next state
            state = next_state

        rewards_per_episode.append(total_reward)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Display progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward}")

    return rewards_per_episode

# Test the Neural Network TD Learning
env = gym.make('CartPole-v1')
rewards = td_neural_network(env)

# Plot the learning curve for Neural Network TD Learning
plt.plot(rewards)
plt.title("TD Learning with Neural Network")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
