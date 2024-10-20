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
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Example values
EXAMPLE_EPSILON = 1.0
EXAMPLE_EPSILON_MIN = 0.01
EXAMPLE_EPSILON_DECAY = 0.995
EXAMPLE_DISCOUNT=0.9
EXAMPLE_ALPHA=0.1
EXAMPLE_TIMESTEP = 100
EXAMPLE_BIN_SIZE = 30
EXAMPLE_EPISODES = 5000
EXAMPLE_EPSILON_MIN = 0.01
EXAMPLE_NUM_SAMPLES = 5000

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

def q_learning(
    q_table, 
    bins, 
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP, epsilon=EXAMPLE_EPSILON,
    epsilon_start=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN
):
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

def initialize_policy(state_space=len(env.observation_space.low),
                      action_space=env.action_space.n,
                      bin_size=EXAMPLE_BIN_SIZE):
    """
    Initializes Policy, randomly.
    --------------------------------
    INPUT:
        state_space: (int)
        action_space: (int)
        bin_size: (int)

    OUTPUT:
        policy: (ndarray) Policy table mapping each discrete state to action.
         - Multi-dimensional array where each element represents a state,
                              and the value is a randomly chosen action for that state.
                              The shape of this array is [bin_size, bin_size, bin_size, bin_size],
                              one dimension for each state variable.
    """
    policy = np.random.choice(action_space, size=[bin_size] * state_space)

    return policy

def policy_evaluation(policy, bins, value_func, discount=EXAMPLE_DISCOUNT,
                      alpha=EXAMPLE_ALPHA, episodes=EXAMPLE_EPISODES):
    """
    Policy Evaluation: Computing V_𝞹(s) for arbitrary 𝞹, using TD(0), a
    model-free
    one-step bootstrapping, using only the immediate next state to udpate the
    value estimate, and 
    ----------------------------------------------------
    For each episode:

    Reset Environment: Start from an initial state.
    Discretize State: Convert continuous observations into discrete indices.
    Step Through the Episode:
        Action Selection: Follow the current policy ππ to choose actions.
        Execute Action: Take the action, observe reward and next state.
        Discretize Next State: Convert next continuous state to discrete indices.
        TD(0) Update:
        V(s)←V(s)+α[r+γV(s′)−V(s)]
        Transition to Next State: Update s←s′s←s′.
    ----------------------------------------------------
    INPUT:
        policy: (ndarray) Current policy mapping state to actions
        bins: (list of ndarrays) Discretized bins for state variables
        value_func: (ndarray) Current value function to be updated
        discount: (float) Discount factor
        alpha: (float) Learning rate
        episodes:(int) Number of episodes for evaluation

    OUTPUT:
        value_func: (ndarray) Updated value function
    """
    for episode in range(epsiodes):
        # Reset env for each episode
        continuous_state, _ = env.reset()
        # Discretize state
        current_state = discrete(cotinuous_state, bins)
        done = False

        # Step thru episodes
        while not done:
            # Select action
            action = policy[current_state]
            # Execute
            next_continuous_state, reward, done, booly, empty_dict = \
            env.step(action)
            # Discretize
            next_state = discrete(next_continuous_state, bins)

            # TD(0) Update
            value_func[current_state] += alpha * (reward + discount *
                                                  value_func[next_state] -
                                                  value_func[current_state])
            # Transition to next state
            current_state = next_state

        return value_func

def policy_improvement(
    initial_policy,
    bins,
    value_func,
    action_space, 
    bin_size,
    dicount=EXAMPLE_DISCOUNT,
    num_samples=EXAMPLE_NUM_SAMPLES):
    """
    Policy Improvement.
    After estimating v_𝞹(s), update policy from 𝞹 to 𝞹' by making it GREEDY
    w.r.t v_𝞹(s):
    - Iterate over all discrete states
    - Select best action
    - Check for policy stabality
    -----------------------------------
    Action Value Estimation:

    For each action aa, Q(s,a)Q(s,a) is approximated as:
    Q(s,a)≈R(s,a)+γV(s′)
    Q(s,a)≈R(s,a)+γV(s′)
        Reward R(s,a)R(s,a): In CartPole, it's typically +1 per timestep.
        Value V(s′)V(s′): Current estimate from Policy Evaluation.

    Selecting Best Action:
        np.argmax(action_values) selects the action with the highest estimated Q(s,a)Q(s,a).

    Policy Update:
        If the best action differs from the current policy's action, update the policy and flag that the policy has changed.
    -----------------------------------
    INPUT:
        initial_policy: Current policy to update
        value_func: (ndarray) 

    OUTPUT:
        policy, policy_stable: (tuple of ?)
    """
    policy = np.copy(initial_policy)
    policy_stable = True

    # Iterate over all discrete states using Cartesian Product ?
    for state in itertools.product(range(bin_size),
                                   repeat=len(env.observation_space.low)):
        state = tuple(state)
        prev_action = policy[state]
        
        action_values = np.zeros(action_space)
        
        for action in range(action_space):
            # Approx. Q(s, a) using current value func, where next state is
            # assumed to be similar to current state... maybe a LIMITATION ?!
            action_values[action] = 1 + discount * value_func[state]

        # Choose best action
        best_action = np.argmax(action_values)

        if best_action != prev_action:
            policy_stable = False
            policy[state] = best_action

    return policy, policy_stable

def stochastic_policy_improvement(
    initial_policy,
    bins,
    value_func,
    bin_size,
    action_space,
    discount=EXAMPLE_DISCOUNT,
    num_sample=EXAMPLE_NUM_SAMPLES
):
    """
    HANDLING MULTIPLE OPTIMAL ACTIONS (TIES)
    If multiple actions yield the same maximum Q(s,a)Q(s,a), the policy can remain 
    stochastic by assigning probabilities to all optimal actions.
    ----------------------------------------------------------
    INPUT: ?

    OUTPUT:
    """
    policy = np.copy(initial_policy)
    policy_stable = True

    # Iterate over all discrete states using Cartesian Product ?
    # Generates all combinations of quintuples (0,0,0,0), (0,0,0,1), ..., (29,29,29,29)
    for state in itertools.product(range(bin_size),
                                   repeat=len(env.observation_space.low)):
        state = tuple(state)
        prev_action = policy[state]
        
        action_values = np.zeros(action_space)
        
        for action in range(action_space):
            # Approx. Q(s, a) using current value func
            q_sa = 1 + discount * value_func[state]
            action_values[action] = q_sa
        
        max_q = np.max(action_values)
        # Find all actions that achieve max. Q(s,a)
        greedy_actions = np.where(action_values == max_q)[0]

        # Selecting greedy action randomly if there are TIES
        if len(greedy_actions) == 1:
            best_action = greedy_actions[0]

        else:
            # Assign equal prob. to all actions
            best_action = np.random.choice(greedy_actions)

        # Test for convergence
        if best_action != prev_action:
            policy_stable = False
            policy[state] = best_action

    return policy, stable_policy

def sarsa(
    q_table,
    bins,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP, 
    epsilon_start=EXAMPLE_EPSILON,
    epsilon_end=EXAMPLE_EPSILON_MIN
):
    """
    SARSA - State Action Reward State2 Action2
    ---------------------------------------------
    On-Policy Temporal Difference (TD) learning for estimating action-value
    functions, using 
    ---------------------------------------------
    INPUT:
        q_table: (ndarray) 
        bins
        episodes=EXAMPLE_EPISODES
        discount=EXAMPLE_DISCOUNT
        alpha=EXAMPLE_ALPHA
        timestep=EXAMPLE_TIMESTEP 
        epsilon_start=EXAMPLE_EPSILON
        epsilon=EXAMPLE_EPSILON
        epsilon_end=EXAMPLE_EPSILON_MIN
    
    OUTPUT:
        episode_rewards: ()
        q_table: (ndarray) Updated Q-table
    """
    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(1, episodes+1):
        # Reset env; initialize state
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)
        
        # Choose action from state from Q using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[current_state])

        total_reward = 0
        done = False

        # For each episode
        while not done:
            # Take action, obseving reward/state
            next_continuous_state, reward, done, booly, empty_dict = \
            env.step(action)
            next_state = discrete(next_continuous_state, bins)
            
            # Choose next action using epsilon-greedy policy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()

            else:
                next_action = np.argmax(q_table[next_state])

            # Update Q-value using SARSA update rule
            td_target = reward + discount * q_table[next_state + (next_action, )]
            td_error = td_target - q_table[current_state + (action, )]
            q_table[current_state + (action, )] += alpha * td_error

            # Update state/action
            current_state = next_state
            action = next_action
            total_reward += reward

        # Decay epsilon: Linear decay
#        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start -
                                                     epsilon_end) * episode /
                       epsilion_decay_episodes)
        episode_rewards.append(total_reward)

        # Output progress
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])

            print(dedent(
                f"""
SARSA - Episode: {episode:5d}, Average Rewards: {avg_reward:6.2f}, Epsilon: {epsilon:.4f}
            """))
        
    env.close()

    return episode_rewards, q_table

def improved_sarsa_delete(
    q_table,
    bins,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha_start=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP,
    epsilon_start=EXAMPLE_EPSILON,
    epsilon_decay=0.9995,  # Slower decay
    epsilon_min=EXAMPLE_EPSILON_MIN,
    max_steps=500  # Maximum steps per episode
):
    """
    Improved SARSA - State Action Reward State Action
    -------------------------------------------------
    On-Policy Temporal Difference (TD) learning for estimating action-value
    functions with improvements for better stability and exploration.

    INPUT:
        q_table: (ndarray) Initial Q-table
        bins: (list of ndarrays) Discretized bins for state variables
        episodes: (int) Number of episodes to train
        discount: (float) Discount factor
        alpha_start: (float) Initial learning rate
        timestep: (int) Interval for reporting training progress
        epsilon_start: (float) Initial exploration rate
        epsilon_decay: (float) Rate at which epsilon decays
        epsilon_min: (float) Minimum value of epsilon
        max_steps: (int) Maximum number of steps per episode

    OUTPUT:
        episode_rewards: (list) Total reward per episode
        q_table: (ndarray) Updated Q-table
    """
    episode_rewards = []
    epsilon = epsilon_start
    alpha = alpha_start

    for episode in range(1, episodes + 1):
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            next_continuous_state, reward, done, _, _ = env.step(action)
            next_state = discrete(next_continuous_state, bins)

            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            td_target = reward + discount * q_table[next_state + (next_action,)]
            td_error = td_target - q_table[current_state + (action,)]
            q_table[current_state + (action,)] += alpha * td_error

            current_state = next_state
            action = next_action
            total_reward += reward
            step += 1

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        alpha = alpha_start * (1 - episode / episodes)  # Linear learning rate decay
        episode_rewards.append(total_reward)

        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"SARSA - Episode: {episode:5d}, Average Reward: {avg_reward:6.2f}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}")

    env.close()

    return episode_rewards, q_table

def delete_sarsa(
    q_table,
    bins,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP, 
    epsilon_start=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN
):
    """
    SARSA - State Action Reward State Action
    ----------------------------------------
    On-Policy Temporal Difference (TD) learning for estimating action-value
    functions.
    
    INPUT:
        q_table: (ndarray) Initial Q-table
        bins: (list of ndarrays) Discretized bins for state variables
        episodes: (int) Number of episodes to train
        discount: (float) Discount factor
        alpha: (float) Learning rate
        timestep: (int) Interval for reporting training progress
        epsilon_start: (float) Initial exploration rate
        epsilon_decay: (float) Rate at which epsilon decays
        epsilon_min: (float) Minimum value of epsilon
    
    OUTPUT:
        episode_rewards: (list) Total reward per episode
        q_table: (ndarray) Updated Q-table
    """
    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(1, episodes + 1):
        # Reset environment and initialize state
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)
        
        # Choose initial action using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        total_reward = 0
        done = False

        while not done:
            # Take action, observe reward and next state
            next_continuous_state, reward, done, _, _ = env.step(action)
            next_state = discrete(next_continuous_state, bins)
            
            # Choose next action using epsilon-greedy policy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            # Update Q-value using SARSA update rule
            td_target = reward + discount * q_table[next_state + (next_action,)]
            td_error = td_target - q_table[current_state + (action,)]
            q_table[current_state + (action,)] += alpha * td_error

            # Update state and action
            current_state = next_state
            action = next_action
            total_reward += reward

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)

        # Output progress
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"SARSA - Episode: {episode:5d}, Average Reward: {avg_reward:6.2f}, Epsilon: {epsilon:.4f}")
    
    env.close()

    return episode_rewards, q_table

def monte_carlo(env, episodes, discount):
    pass

def performance_evalutation(rewaeds, algorithm_name):
    pass

def delete_main():
    # Initialize Q-table and bins
    q_table, bins = Qtable(bin_size=EXAMPLE_BIN_SIZE)

    # Q-Learning
    q_learning_rewards, q_learning_q_table = q_learning(
        q_table=np.copy(q_table),
        bins=bins,
        episodes=EXAMPLE_EPISODES,
        discount=EXAMPLE_DISCOUNT,
        alpha=EXAMPLE_ALPHA,
        timestep=EXAMPLE_TIMESTEP,
        epsilon=EXAMPLE_EPSILON,
        epsilon_decay=EXAMPLE_EPSILON_DECAY,
        epsilon_min=EXAMPLE_EPSILON_MIN
    )
    performance_evaluation(q_learning_rewards, 'Q-Learning')

    # SARSA
    sarsa_rewards, sarsa_q_table = sarsa(
        q_table=np.copy(q_table),
        bins=bins,
        episodes=EXAMPLE_EPISODES,
        discount=EXAMPLE_DISCOUNT,
        alpha=EXAMPLE_ALPHA,
        timestep=EXAMPLE_TIMESTEP,
        epsilon=EXAMPLE_EPSILON,
        epsilon_decay=EXAMPLE_EPSILON_DECAY,
        epsilon_min=EXAMPLE_EPSILON_MIN
    )
    performance_evaluation(sarsa_rewards, 'SARSA')

    # Monte Carlo
    monte_carlo_rewards, monte_carlo_q_table = monte_carlo(
        q_table=np.copy(q_table),
        bins=bins,
        episodes=EXAMPLE_EPISODES,
        discount=EXAMPLE_DISCOUNT,
        epsilon=EXAMPLE_EPSILON,
        epsilon_decay=EXAMPLE_EPSILON_DECAY,
        epsilon_min=EXAMPLE_EPSILON_MIN
    )
    performance_evaluation(monte_carlo_rewards, 'Monte Carlo')

    # Note: Policy Iteration may not be effective due to the limitations mentioned earlier.

#    results = {
#        "Policy Iteration":,
#        "SARSA": ,
#        "Q-Learning": ,
#        "Monte Carlo":
#    }

q_table, bins = Qtable()
thing = improved_sarsa_delete(q_table, bins)
#breakpoint()
