import sys
import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque
import Policy_Gradient_REINFORCE

# Create and wrap the environment
env = gym.make('InvertedPendulum-v4', render_mode='human')

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

# Constants
num_episodes = sys.maxsize
max_timesteps = 100000
statistics_window_size = 50

rewards_moving_window = deque(maxlen=statistics_window_size)

def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    return state

agent = Policy_Gradient_REINFORCE.Policy_Gradient_REINFORCE(action_space_size=action_space_dims, state_space_size=obs_space_dims)
for episode in range(num_episodes):
    state = env.reset(seed=0)[0]
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
            rewards_moving_window.append(np.sum(episode_rewards))
            print("Episode:", episode, "Average Reward:", np.mean(rewards_moving_window))
            break
            pass
        pass
    # Train once for every episode because all the rewards for that episode are needed
    agent.train_network()
    pass