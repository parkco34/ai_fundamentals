#!/usr/bin/env python
def q_learning_linear(
    theta,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    epsilon=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN
):
    episode_rewards = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Choose action with max Q-value
                q_values = []
                for a in range(env.action_space.n):
                    features = get_features(state, a)
                    q_value = np.dot(theta, features)
                    q_values.append(q_value)
                action = np.argmax(q_values)

            next_state, reward, done, _, _ = env.step(action)

            # Compute TD target
            q_values_next = []
            for a in range(env.action_space.n):
                features_next = get_features(next_state, a)
                q_value_next = np.dot(theta, features_next)
                q_values_next.append(q_value_next)
            td_target = reward + discount * np.max(q_values_next)

            # Compute TD error
            features_current = get_features(state, action)
            q_value_current = np.dot(theta, features_current)
            td_error = td_target - q_value_current

            # Update theta
            theta += alpha * td_error * features_current

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        # Optional: Print progress
        if episode % EXAMPLE_TIMESTEP == 0:
            avg_reward = np.mean(episode_rewards[-EXAMPLE_TIMESTEP:])
            print(f"Linear Q-Learning -> Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    return episode_rewards, theta
