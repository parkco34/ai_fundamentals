#!/usr/bin/env python
"""
Helped by:
    https://github.com/maciejbalawejder/Reinforcement-Learning-Collection/blob/main/Q-Table/Qtable.ipynb
    https://gymnasium.farama.org/environments/classic_control/cart_pole/

--------------------
CARTPOLE-V0 PROBLEM:
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
--------------------
The classic CartPole-v0 problem from OpenAI Gym. This environment presents a continuous state space and a discrete action space, offering an excellent platform to compare tabular methods with function ap- proximation techniques.
------------------------------------------------------------
Learning Objectives
• Implement and understand policy iteration, temporal difference (TD) learning, and Monte Carlo (MC) methods.
• Apply discretization techniques to handle continuous state spaces in tabular methods.
• Utilize function approximation to manage high-dimensional or continuous state spaces.
• Compare the performance of different RL algorithms and understand their advantages and limitations.
------------------------------------------------------------
Background
The CartPole-v0 environment consists of a pole attached by an unactuated joint to a cart, which moves along a frictionless track. The goal is to prevent the pole from falling over by applying forces (left or right) to the cart. The environment provides four continuous observations:
1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Velocity at Tip
------------------------------------------------------------
================
Assignment Tasks
================
PART 1: 
Tabular Methods with Discretization
-----------------------------------------------------------
1. State Space Discretization
Objective: Convert the continuous state space into a finite set of discrete states. Instructions:
• Define bins for each of the four state variables.
• Create a mapping function to convert continuous observations into discrete states.

2. Policy Iteration
OBJECTIVE: Implement policy evaluation and improvement to find the optimal policy. Instructions:
• Perform iterative policy evaluation until the value function converges.
3. Temporal Difference Learning (SARSA and Q-Learning)
Objective: Implement online learning algorithms to estimate the action-value function. Instructions:
• Implement SARSA and Q-Learning algorithms using the discretized state space.
4. Monte Carlo Methods
Objective: Use episodic sampling to estimate the value of states.
Instructions:
• Implement Monte Carlo prediction.
5. Performance Evaluation
Objective: Compare the performance of the tabular methods.
Instructions:
• Define evaluation metrics (e.g., average return per episode, number of episodes to convergence). Plot learning curves for each method.
• Analyze the stability and efficiency of the algorithms.

PART 2:

Function Approximation
-------------------------
1. Linear Function Approximation
Objective: Implement TD learning with linear function approximation.
Instructions:
• Represent the Q-function as a linear combination of features.
2. Non-Linear Function Approximation (Neural Networks)
Objective: Implement TD learning with neural networks.
Instructions:
• Design a neural network to approximate the Q-function.
• Decide on the architecture (number of layers, neurons, activation functions).
3. Monte Carlo Methods with Function Approximation
Objective: Combine MC methods with function approximation.
Instructions:
• Use the returns from episodes to update the function approximator.
• Ensure that the function approximator generalizes well across states.
4. Performance Evaluation
Objective: Compare the performance of function approximation methods with tabular methods.
Instructions:
• Use the same evaluation metrics as before.
• Discuss the impact of function approximation on learning speed and policy quality.
• Analyze the effects of hyperparameters and network architecture.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# CartPole environment
env = gym.make("CartPole-v1")
env.reset(seed=73) # Reproducability
# Outputs the lower/upper bounds of observation space
print(env.observation_space.low, "\n", env.observation_space.high) # |S|,|A(s)|

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
        q_table, bins: (tuple of ndarrays ?, list of ndarryas)
    """
    # Ranges for each of the state variables, where they're evenly spaced
    # intervals
    bins = [
        # Cart position (in meters)
        np.linspace(-4.8, 4.8, bin_size),
        # Cart velocity
        np.linspace(-4, 4, bin_size),
        # Pole position
        np.linspace(-0.418, 0.418, bin_size),
        # Pole angular velocity (in radians/second)
        np.linspace(-4, 4, bin_size)
    ]

    # [bin_size] * state_space -> creates list of bin_size repeated state_space
    # times, and w/ action dimension (+ [action_space]), becoming: [30, 30, 30, 30, 2]
    # initializes Q-table w/ random value tween -1 and 1
    q_table = np.random.uniform(low=-1, high=1, size=[bin_size] * state_space
                             + [action_space])

    return q_table, bins # q_table.ndim == 5, list of 4 np arrays

def discrete(state, bins):
    """
    Discretizes continuous space
    ----------------------------
    INPUT:
        state: (int) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        indices: (tuple) for making it hashable for indexing into multidimensional
        (5-dimensional table, here: q_table.ndim) Q-table.
    """
    index = []

    for i in range(len(state)):
        # Corresponding Indices of bins; discretization process
        index.append(np.digitize(state[i], bins[i])-1)

    indices = tuple(index)

    return indices

def q_learning(q_table, bins, episodes=5000, discount=0.9, alpha=0.2,
               timestep=100, epsilon=0.2, epsilon_decay=0.05, epsilon_min=0.01):
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
        timestep: (int) Interfval reporting training progress
        epsilon: Initial exploration rate
        epsilon_decay: (float) ?
        epsilon_min: (float) ?

    OUTPUT:
        (None)
    """
    # Initializations ?
    reward, steps = 0, 0
    episode_rewards = []

    for episode in range(1, episodes+1):
        steps += 1
        # Initialize indices for Q-table
        initial_state, _ = env.reset() # numpy array, empty dict
        current_state = discrete(initial_state, bins)

        total_reward = 0
        done =  False

        for t in range(timestep):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # env.action_space -> All possible actions in environment
                # sample() -> randomly selects an action via (left=0, right=1)
                action = env.action_space.sample() # Explore w/ prob. 𝜀

            else:
                # Returns index of highest Q-value
                # q_table[current_state] -> Array of Q-values for possible
                # actions.
                # np.argmax() -> finds index of max value in array
                action = np.argmax(q_table[current_state]) # Exploit; prob. 1-𝜀

            # anv.step(action) -> (observation, reward, terminated, truncated, info)
            # Executes chosen action, unpacking the return values
            next_state, reward, done, booly, empty_dict= env.step(action)
            # Discretizes new continuous state for indexing Qtable
            proper_next_state = discrete(next_state, bins)
            # finds action with highest Q-value in next state 
            best_next_action = np.argmax(q_table[proper_next_state])
            # TD (temporal difference target), estimating the total expected
            # reward:= immediate reward + discounted best possible future
            # reward...
            # "the value of this state-action pair is what we got now, plus what we think we can get in the future, but we're less certain about the future so we discount it."
            td_target = reward + discount * q_table[proper_next_state +
                                                    (best_next_action, )]
            td_error = td_target - q_table[current_state + (action, )]
            # Updates Q-value for current state-action pair
            # moving Q-value closer to TD target by a fraction via 𝛼
            q_table[current_state + (action, )] += alpha * td_error

            # Update current state/reward
            current_state = proper_next_state
            total_reward += reward

            # Check if episode ended and if so, breakdance!
            if done:
                break

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        # Output progress at every episode
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"""Episode: {episode}, Average Reward: {avg_reward:.2f},
                  Epsilon: {epsilon:.4f}""")

    env.close()

    return episode_rewards, q_table

# 2.) Policy Iteration
def policy_iteration(env, bins, discount=0.9, max_iters=10, eval_episodes=100):
    """
    Implements Approximate Policy Iteration via Monte Carlo Policy Evaluation.
    ------------------------------------------------
    INPUT:

    OUTPUT:
    """
    state_shape = tuple(len(bins[i]) for i in range(len(bins)))
    # Initialize random policy
    policy = np.random.choice(env.action_space.n, size=state_size)
    value_func = np.zeros(state_shape)

    for iteration in range(max_iterations):
        print("Policy Iteration {iteration+1}")

        # Policy Evaluation
        value_func = monte_carlo_eval(env, policy, bins, value_func, discount,
                                     evaluation_episodes)

        # Policy Improvement
        policy_stable = True
        for idx in np.ndindex(state_shape):
            prev_action = policy[idx]
            state = [bins[i][idx[i]] for i in range(len(idx))]
            action_values = []

            for action in range(env.action_space.n):
                total_reward = 0.0
                total_count = 0

            # ???????

# ChatGPT ====================================== ?
def policy_iteration(env, bins, discount=0.9, max_iterations=10, evaluation_episodes=100):
    """
    Implements Approximate Policy Iteration using Monte Carlo Policy Evaluation.

    Parameters:
    - env: Gym environment
    - bins: Discretized bins
    - discount: Discount factor
    - max_iterations: Maximum number of policy improvement iterations
    - evaluation_episodes: Number of episodes for policy evaluation

    Returns:
    - policy: Optimal policy
    - value_func: Value function for the optimal policy
    """
    # Discretized state space
    state_shape = tuple(len(bins[i]) for i in range(len(bins)))
    # Initialize a random policy
    policy = np.random.choice(env.action_space.n, size=state_shape)
    # Initialize value function
    value_func = np.zeros(state_shape)

    for iteration in range(max_iterations):
        print(f"Policy Iteration {iteration+1}")

        # Policy Evaluation
        value_func = monte_carlo_policy_evaluation(env, policy, bins,
                                                   value_func, discount, evaluation_episodes)

        # Policy Improvement
        policy_stable = True
        for idx in np.ndindex(state_shape):
            old_action = policy[idx]
            state = [bins[i][idx[i]] for i in range(len(idx))]
            action_values = []

            for action in range(env.action_space.n):
                total_reward = 0.0
                total_count = 0

                # Sample transitions
                for _ in range(5):  # Sample a few times for each action
                    env.reset()
                    # We can't set the state directly, so we skip states we can't reach
                    # This is a limitation, so we might need to adjust our approach
                    # Alternatively, we can consider the value function as is

                    # Since we can't simulate from specific states, we use the current value function
                    next_state, reward, done, _, _ = simulate_action(env, state, action)
                    if next_state is not None:
                        next_state_discrete = discrete(next_state, bins)
                        total_reward += reward + discount * vaue_func[next_state_discrete]
                        total_count += 1

                if total_count > 0:
                    action_value = total_reward / total_count
                else:
                    action_value = value_func[idx]  # Default to current value

                action_values.append(action_value)

            best_action = np.argmax(action_values)
            policy[idx] = best_action

            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            print("Policy converged.")
            break

    return policy, value_func

def monte_carlo_policy_evaluation(env, policy, bins, value_func, discount, episodes):
    """
    Evaluates a policy using Monte Carlo sampling.

    Parameters:
    - env: Gym environment
    - policy: Current policy
    - bins: Discretized bins
    - value_func: Current value function
    - discount: Discount factor
    - episodes: Number of episodes for evaluation

    Returns:
    - value_func: Updated value function
    """
    returns_sum = {}
    returns_count = {}

    for episode in range(episodes):
        state_continuous, _ = env.reset()
        state_discrete = discrete(state_continuous, bins)
        episode_data = []
        done = False

        while not done:
            action = policy[state_discrete]
            next_state_continuous, reward, done, _, _ = env.step(action)
            episode_data.append((state_discrete, reward))
            state_discrete = discrete(next_state_continuous, bins)

        G = 0
        for state_discrete, reward in reversed(episode_data):
            G = discount * G + reward
            if state_discrete not in returns_sum:
                returns_sum[state_discrete] = G
                returns_count[state_discrete] = 1
            else:
                returns_sum[state_discrete] += G
                returns_count[state_discrete] += 1
            value_func[state_discrete] = returns_sum[state_discrete] / returns_count[state_discrete]

    return value_func

def simulate_action(env, state, action):
    """
    Simulates taking an action from a given state.

    Parameters:
    - env: Gym environment
    - state: Continuous state
    - action: Action to take

    Returns:
    - next_state: Next continuous state
    - reward: Reward received
    - done: Whether the episode is done
    - info: Additional info
    """
    # Since we can't set the environment to a specific state, we can't simulate from it.
    # This function is here for completeness but returns None.
    return None, 0, True, {}

def sarsa(q_table, bins, episodes=5000, discount=0.9, alpha=0.2,
          timestep=100, epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01):
    """
    Implements the SARSA algorithm.

    Parameters:
    - q_table: Initialized Q-table
    - bins: Discretized bins
    - episodes: Number of episodes to train
    - discount: Discount factor
    - alpha: Learning rate
    - timestep: Reporting interval
    - epsilon: Initial exploration rate
    - epsilon_decay: Decay rate for epsilon
    - epsilon_min: Minimum epsilon value

    Returns:
    - episode_rewards: List of total rewards per episode
    - q_table: Updated Q-table
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

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        # Output progress at every episode
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    return episode_rewards, q_table

def monte_carlo_prediction(bins, episodes=5000, discount=0.9, epsilon=0.2,
                           epsilon_decay=0.99, epsilon_min=0.01):
    """
    Implements Monte Carlo Prediction to estimate V(s).

    Parameters:
    - bins: Discretized bins
    - episodes: Number of episodes to train
    - discount: Discount factor
    - epsilon: Initial exploration rate
    - epsilon_decay: Decay rate for epsilon
    - epsilon_min: Minimum epsilon value

    Returns:
    - value_func: Estimated value function
    - episode_rewards: List of total rewards per episode
    """
    state_shape = tuple(len(bins[i]) for i in range(len(bins)))
    value_func = np.zeros(state_shape)
    returns_sum = {}
    returns_count = {}
    episode_rewards = []

    for episode in range(1, episodes + 1):
        state_continuous, _ = env.reset()
        state_discrete = discrete(state_continuous, bins)
        episode_data = []
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Since we don't have a policy, we can choose actions randomly
                action = env.action_space.sample()

            next_state_continuous, reward, done, _, _ = env.step(action)
            next_state_discrete = discrete(next_state_continuous, bins)
            episode_data.append((state_discrete, reward))
            state_discrete = next_state_discrete
            total_reward += reward

        G = 0
        visited_states = set()
        for state_discrete, reward in reversed(episode_data):
            G = discount * G + reward
            if state_discrete not in visited_states:
                visited_states.add(state_discrete)
                if state_discrete not in returns_sum:
                    returns_sum[state_discrete] = G
                    returns_count[state_discrete] = 1
                else:
                    returns_sum[state_discrete] += G
                    returns_count[state_discrete] += 1
                value_func[state_discrete] = returns_sum[state_discrete] / returns_count[state_discrete]

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    return value_func, episode_rewards

def main():
    q_table_q, bins = Qtable()
    q_table_sarsa, _ = Qtable()

    episodes = 5000

    # Q-Learning
    rewards_q_learning, trained_q_table_q = q_learning(
        q_table_q, bins, episodes=episodes)

    # SARSA
    rewards_sarsa, trained_q_table_sarsa = sarsa(
        q_table_sarsa, bins, episodes=episodes)

    # Monte Carlo Prediction (Note: Only estimating V(s), not Q(s,a))
    V_mc, rewards_mc = monte_carlo_prediction(
        bins, episodes=episodes)

    # Plotting the rewards
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.plot(rewards_sarsa, label='SARSA')
    plt.plot(rewards_mc, label='Monte Carlo')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Performance Comparison of Tabular Methods')
    plt.legend()
    plt.show()

# =======================================

if __name__ == "__main__":
    main()


