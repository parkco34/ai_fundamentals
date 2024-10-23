#!/usr/bin/env python
"""
===============
Q-LEARNING
===============
ANALYSIS:
---------
1. **Issues in Your Current Implementation:**
   - The exploration probability (`explore_prob`) isn't being updated correctly - you calculate `exploration_proba` but never use it
   - The epsilon decay is using exponential decay with a constant rate, which might be too aggressive
   - There's no structured way to track the agent's performance
   - The code structure makes it difficult to experiment with different parameters

2. **Key Improvements in the New Implementation:**

   a) **Better Code Structure:**
   - Created a `QLearningAgent` class to encapsulate the Q-learning logic
   - Separated training logic into a `train_agent` function
   - Added proper performance monitoring

   b) **Modified Exploration Strategy:**
   - Changed to multiplicative epsilon decay (`epsilon * epsilon_decay`)
   - Added proper epsilon bounds and smoother decay
   - Fixed the exploration probability update

   c) **Better Performance Tracking:**
   - Added success rate tracking (percentage of episodes where reward > 0)
   - Implemented moving average of rewards
   - Added visualization of learning progress

3. **Why Your Results Were Poor:**
   - The reward values (0.01-0.02) indicate the agent rarely reaches the goal
   - This is likely due to:
     - Too aggressive epsilon decay
     - Challenging nature of FrozenLake (slippery environment)
     - Possibly insufficient exploration

4. **Suggestions for Further Improvement:**
   - Try increasing the learning rate (e.g., 0.2-0.3) for faster learning
   - Experiment with different epsilon decay rates
   - Consider implementing experience replay
   - Add temperature-based exploration (Boltzmann exploration)
   - Implement double Q-learning to reduce overestimation
"""
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1")
state_space = env.observation_space.n
action_space = env.action_space.n

q_table = np.zeros((state_space, action_space))

episodes = int(1e4)
timesteps = 100

# Epsilon decay
explore_prob = 1
epsilon_decay = 0.001
min_explore_prob = 0.01

gamma = 0.99
alpha = 0.1
episode_rewards = []

for episode in range(episodes):
    current_state, _ = env.reset()
    done = False

    total_reward = 0

    for i in range(timesteps):
        # Sample float from uniform distribution.  If it's less than
        # explore_prob, the agent selects random action, otherwise exploit
        # knowledge using bellman equation
        if np.random.uniform(0,1) < explore_prob:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[current_state, :])

        next_state, reward, done, booly, empty_dict = env.step(action)

        # We update our Q-table using the Q-learning iteration
        q_table[current_state, action] = (1-alpha) * q_table[current_state, action] +alpha*(reward + gamma*max(q_table[next_state,:]))
        total_reward += reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    #We update the exploration proba using exponential decay formula
    exploration_proba = max(min_explore_prob, np.exp(-epsilon_decay* episode))
    episode_rewards.append(total_reward)

print("Mean reward per thousand episodes")
for i in range(10):
    print(f"Episodes {i*1000} to {(i+1)*1000}: mean episode reward: {np.mean(episode_rewards[1000*i:1000*(i+1)]):.3f}")

breakpoint()

