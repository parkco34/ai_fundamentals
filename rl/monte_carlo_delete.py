#!/usr/bin/env python
def monte_carlo(
    q_table,
    bins,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    epsilon=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN
):
    """
    First-Visit Monte Carlo Control with Exploring Starts.
    """
    returns_sum = {}
    returns_count = {}
    episode_rewards = []

    for episode in range(1, episodes + 1):
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)
        episode_memory = []
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])

            next_state_continuous, reward, done, _, _ = env.step(action)
            next_state = discrete(next_state_continuous, bins)
            episode_memory.append((current_state, action, reward))
            total_reward += reward
            current_state = next_state

        # Calculate returns and update Q-values
        G = 0
        visited_state_actions = set()
        for state, action, reward in reversed(episode_memory):
            G = discount * G + reward
            sa_pair = (state, action)
            if sa_pair not in visited_state_actions:
                visited_state_actions.add(sa_pair)
                if sa_pair not in returns_sum:
                    returns_sum[sa_pair] = 0.0
                    returns_count[sa_pair] = 0
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1
                q_table[state + (action,)] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        # Output progress
        if episode % EXAMPLE_TIMESTEP == 0:
            avg_reward = np.mean(episode_rewards[-EXAMPLE_TIMESTEP:])
            print(f"Monte Carlo - Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    return episode_rewards, q_table


