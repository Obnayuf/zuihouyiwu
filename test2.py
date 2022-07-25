import gym
import sys
import random
import torch
import numpy as np
import itertools
from collections import namedtuple, deque
import matplotlib.pyplot as plt


# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal


# python imports
from operator import itemgetter
import time


# Use GPU is possible else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('LunarLanderContinuous-v2')

def init_seed(seed):
    #run this before any agent. checked to stabilize the randomness.
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCriticContinuous, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.ReLU()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPOContinuousAgent:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.time_step = 0
        self.memory = Memory()

        self.policy = ActorCriticContinuous(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCriticContinuous(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, self.memory).cpu().data.numpy().flatten()

    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def step(self, reward, done):
        self.time_step += 1
        # Saving reward and is_terminals:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if self.time_step % 1200 == 0:
            self.update()
            self.memory.clear_memory()
            self.time_step = 0

    def act(self, state):
        return self.select_action(state)


def train(agent, n_episodes=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, agent_type="DQN"):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        agent_type (str): determines agent's type (q-learning , sac)
    """
    scores = []  # list containing scores from each episode
    avg_scores = []  # list contating avg scores
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # choose an action
            if agent_type == "PPO_CONTINUOUS":
                action = agent.act(state)
            # do action in environment
            next_state, reward, done, _ = env.step(action)

            # observe and learn (by the agent)
            if agent_type == "PPO_DISCRETE" or "PPO_CONTINUOUS":
                agent.step(reward, done)
            # accumulate score and move to next state
            state = next_state
            score += reward

            # stop episode if done
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        avg_scores.append(np.mean(scores_window))  # save current avg score


        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break

    return scores, avg_scores

def main():

    ppo_scores = []
    ppo_avg_scores = []
    ppo_times = []

    solved_reward = 200  # stop training if avg_reward > solved_reward
    log_interval = 1  # print avg reward in the interval
    max_episodes = 2000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode

    update_timestep = 1200  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 15  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = 1
    RUNS = 3
    for i in range(1, RUNS + 1):
        init_seed(i)
        PPO = PPOContinuousAgent(8, 2, action_std, lr, betas, gamma, K_epochs, eps_clip)
        start = time.time()
        score, avg_score = train(PPO, n_episodes=1500, agent_type="PPO_CONTINUOUS")
        end = time.time()
        ppo_scores.append(score)
        ppo_avg_scores.append(avg_score)
        ppo_times.append(end - start)
        torch.save(PPO.policy.state_dict(), "PPO_" + str(i) + ".pt")

if __name__ == '__main__':
    main()