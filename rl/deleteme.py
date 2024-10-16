#!/usr/bin/env python
def q_learning(q_table, bins, episodes=5000, discount=0.9, alpha=0.2,
               timestep=5000, epsilon=0.8, epsilon_decay=0.99,
               epsilon_min=0.01):
    """
    Implements Q-learning algorithm for the CartPole-v0 problem.

    Args:
    q_table (np.array): Initial Q-table
    bins (list): Discretization bins for each state dimension
    episodes (int): Number of episodes to train
    discount (float): Discount factor for future rewards
    alpha (float): Learning rate
    timestep (int): Maximum number of timesteps per episode
    epsilon (float): Initial exploration rate
    epsilon_decay (float): Decay rate for epsilon
    epsilon_min (float): Minimum epsilon value

    Returns:
    np.array: Trained Q-table
    list: Rewards per episode
    """
    env = gym.make("CartPole-v1")
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = discrete(state, bins)
        total_reward = 0
        done = False

        for t in range(timestep):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action and observe new state and reward
            next_state, reward, done, _, _ = env.step(action)
            next_state = discrete(next_state, bins)

            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            # Whjat's this about ?
            td_target = reward + discount * q_table[next_state + (best_next_action,)]
            td_error = td_target - q_table[state + (action,)]
            q_table[state + (action,)] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        rewards_per_episode.append(total_reward)

    env.close()
    return q_table, rewards_per_episode

