#!/usr/bin/env python
def policy_iteration(bins, max_iterations=100, eval_episodes=1000, theta=1e-4):
    """
    Complete policy iteration algorithm combining evaluation and improvement.
    """
    # Initialize random policy and value function
    policy = np.random.randint(0, env.action_space.n, size=[len(bin) for bin in bins])
    value_func = np.zeros_like(policy, dtype=float)

    for iteration in range(max_iterations):
        # 1. Policy Evaluation
        value_func = policy_evaluation(policy, bins, value_func)

        # 2. Policy Improvement
        new_policy = policy_improvement(value_func, bins)

        # Check convergence
        if np.array_equal(policy, new_policy):
            print(f"Policy converged after {iteration + 1} iterations")
            break

        policy = new_policy

    return policy, value_func

def policy_evaluation(policy, bins, value_func, episodes=1000):
    """
    Evaluates policy using iterative updates.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            discrete_state = discrete(state, bins)
            action = policy[discrete_state]
            next_state, reward, done, _, _ = env.step(action)
            next_discrete = discrete(next_state, bins)

            # Value function update
            value_func[discrete_state] += alpha * (
                reward +
                discount * value_func[next_discrete] -
                value_func[discrete_state]
            )

            state = next_state

    return value_func

def policy_improvement(value_func, bins):
    """
    Makes policy greedy with respect to value function.
    """
    new_policy = np.zeros_like(value_func, dtype=int)

    for state in np.ndindex(value_func.shape):
        action_values = np.zeros(env.action_space.n)

        # Evaluate each action
        for action in range(env.action_space.n):
            next_states_values = []
            state_coords = [bins[i][state[i]] for i in range(len(state))]

            # Simulate action
            env_copy = env.copy()
            env_copy.state = state_coords
            next_state, reward, _, _, _ = env_copy.step(action)
            next_discrete = discrete(next_state, bins)

            action_values[action] = reward + discount * value_func[next_discrete]

        # Choose best action
        new_policy[state] = np.argmax(action_values)

    return new_policy


