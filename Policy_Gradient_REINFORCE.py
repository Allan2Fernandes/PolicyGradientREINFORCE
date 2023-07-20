import torch
import numpy as np
from PolicyNetwork import PolicyNetwork

class Policy_Gradient_REINFORCE:
    def __init__(self, action_space_size, state_space_size):
        self.gamma = 0.99
        self.policy_network = PolicyNetwork(action_space_size=action_space_size, observation_space_size=state_space_size)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.00008)
        self.episode_rewards = []
        self.log_probs = []
        pass

    def act(self, state):
        means, devs = self.policy_network(state)
        action_distribution = torch.distributions.normal.Normal(means, devs)
        action = action_distribution.sample()
        action = torch.tensor([torch.squeeze(torch.squeeze(action, dim=0))])

        #self.log_probs.append(action_distribution.log_prob(action))
        return action, action_distribution.log_prob(action)

    def train_network(self):
        reward_to_go = 0
        rewards_every_step = []
        for reward in reversed(self.episode_rewards): #Reversed
            reward_to_go = reward + self.gamma*reward_to_go
            rewards_every_step.insert(0, reward_to_go)
            pass



        loss = 0
        for log_prob, step_reward in zip(self.log_probs, rewards_every_step):
            loss += log_prob*step_reward
            pass



        loss = -loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty them every episode
        self.log_probs = []
        self.episode_rewards = []
        pass