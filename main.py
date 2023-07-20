import sys

import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque

import Policy_Gradient_REINFORCE

# Create and wrap the environment
env = gym.make('InvertedPendulum-v4', render_mode='human')
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
num_episodes = sys.maxsize
max_timesteps = 100000


def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    return state

agent = Policy_Gradient_REINFORCE.Policy_Gradient_REINFORCE(action_space_size=action_space_dims, state_space_size=obs_space_dims)
for episode in range(num_episodes):
    state = env.reset()[0]
    time_alive = deque(maxlen=100)
    episode_rewards = []
    for t in range(max_timesteps):
        state = preprocess_state(state)
        action, log_prob = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.episode_rewards.append(reward)
        agent.log_probs.append(log_prob)
        episode_rewards.append(reward)

        state = next_state
        if done or truncated:
            time_alive = t
            #print("Reward for episode {0} = {1}".format(episode, np.mean(episode_rewards)))
            print("Episode {0} time alive = {1}".format(episode, t))

            break
            pass
        pass
    # Train once for every episode because all the rewards for that episode are needed
    agent.train_network()
    pass