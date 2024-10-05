#!/usr/bin/env python
import gym
import numpy as np

env = gym.make('CartPole-v0')
actions = [0, 1]
alpha = 0.1
gamma = 1.0
epsilon = 0.1
num_iterations = 10
num_episodes = 1000

# Initialize policy and value function
policy = {}
value_function = {}
for state in all_possible_states:
    policy[state] = np.random.choice(actions)
    value_function[state] = 0.0

for i in range(num_iterations):
    # Policy Evaluation
    for episode in range(num_episodes):
        observation = env.reset()
        state = discretize_state(observation)
        done = False
        while not done:
            action = policy[state]
            next_observation, reward, done, _ = env.step(action)
            next_state = discretize_state(next_observation)
            td_target = reward + gamma * value_function[next_state] * (not done)
            td_error = td_target - value_function[state]
            value_function[state] += alpha * td_error
            state = next_state
    # Policy Improvement
    for state in value_function.keys():
        q_values = []
        for action in actions:
            env.env.state = state  # Note: This may not work as expected in OpenAI Gym
            next_observation, reward, done, _ = env.step(action)
            next_state = discretize_state(next_observation)
            q_value = reward + gamma * value_function[next_state] * (not done)
            q_values.append(q_value)
        best_action = np.argmax(q_values)
        policy[state] = actions[best_action]

Q = {}
for state in all_possible_states:
    Q[state] = np.zeros(len(actions))

for episode in range(num_episodes):
    observation = env.reset()
    state = discretize_state(observation)
    action = choose_action(Q, state, epsilon)
    done = False
    while not done:
        next_observation, reward, done, _ = env.step(action)
        next_state = discretize_state(next_observation)
        next_action = choose_action(Q, next_state, epsilon)
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] * (not done) - Q[state][action])
        state = next_state
        action = next_action

for episode in range(num_episodes):
    observation = env.reset()
    state = discretize_state(observation)
    done = False
    while not done:
        action = choose_action(Q, state, epsilon)
        next_observation, reward, done, _ = env.step(action)
        next_state = discretize_state(next_observation)
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] * (not done) - Q[state][action])
        state = next_state

def choose_action(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])


V = {}
returns_sum = {}
returns_count = {}
for state in all_possible_states:
    V[state] = 0.0
    returns_sum[state] = 0.0
    returns_count[state] = 0

for episode in range(num_episodes):
    observation = env.reset()
    episode_states = []
    episode_rewards = []
    done = False
    while not done:
        state = discretize_state(observation)
        episode_states.append(state)
        action = np.random.choice(actions)
        observation, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
    G = 0
    visited_states = set()
    for t in reversed(range(len(episode_states))):
        state = episode_states[t]
        G = gamma * G + episode_rewards[t]
        if state not in visited_states:
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]
            visited_states.add(state)


:tanm

