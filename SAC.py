import gym
import sys
import random
import torch
import numpy as np
import itertools
from collections import namedtuple, deque
import matplotlib.pyplot as plt
#%matplotlib inline
from torch.distributions.normal import Normal
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
import physics_sim
# python imports
from operator import itemgetter
import time
import uavutils
import operator

# Use GPU is possible else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = physics_sim.PhysicsSim()
STATE_SIZE = 15  #8
ACTION_SIZE = 4 # 2 - for continuous application

#These parameters assume all actions has the same high-low
ACTION_HIGH = 1#[0]
ACTION_LOW = -1#[0]

# Define the agents hyperparameters, incl. Replay Memory

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.01  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 2  # how often to update the network
reporter = SummaryWriter('runs/SAC/second')
ALPHA = 1
INITIAL_BETA = 0.4

#layer initializations
def init_layer(layer):
    size = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(size)
    return (-lim,lim)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
#copy model parameters from source to target
def copy_params(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def init_seed(seed):
    #run this before any agent. checked to stabilize the randomness.
    #env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, is_action_discrete=True):  # , seed , action_size
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.buffer_size = buffer_size
        self.is_action_discrete = is_action_discrete

        self._index = 0
        self.memory = []
        # self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print (reward,np.array(reward))
        e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        if self._index >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self._index] = e
        self._index = (self._index + 1) % self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        if self.is_action_discrete:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        else:  # action is continuous
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        # print (rewards,np.array[rewards])
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, alpha, is_action_discrete=True):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size, is_action_discrete)
        self.alpha = alpha
        self.max_priority = 1.0
        # self.next_index =  0

        tree_size = 1
        while tree_size < buffer_size:
            tree_size *= 2

        self.sumtree = SumSegmentTree(tree_size)
        self.minsumtree = MinSegmentTree(tree_size)
        # self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        index = self._index
        super().add(state, action, reward, next_state, done)
        # print(self.next_index,self.sumtree._capacity)
        self.sumtree[index] = self.max_priority ** self.alpha
        self.minsumtree[index] = self.max_priority ** self.alpha
        # self.next_index = (self.next_index + 1) % self.buffer_size

    def sample_proportional(self):
        res = []
        p_total = self.sumtree.sum(0, len(self.memory) - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.sumtree.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        indexes = self.sample_proportional()

        weights = []
        p_min = self.minsumtree.min() / self.sumtree.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in indexes:
            p_sample = self.sumtree[idx] / self.sumtree.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.from_numpy(np.array(weights)).float().to(device)

        # print (indexes,weights)
        experiences = list(itemgetter(*indexes)(self.memory))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones, weights, indexes)

    def update_priorities(self, indexes, priorities):
        for index, priority in zip(indexes, priorities):
            self.sumtree[index] = priority ** self.alpha
            self.minsumtree[index] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)





class SoftQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_units=64, init_w=3e-3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # small init to layer weights
        # self.fc1.weight.data.uniform_(*init_layer(self.fc1))
        # self.fc2.weight.data.uniform_(*init_layer(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.apply(init)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        # x = torch.cat((state,action),dim=1)
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=64, init_w=3e-5, log_std_min=-20,
                 log_std_max=2):  # , action_limit
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # self.action_limit = action_limit

        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)

        self.mean_layer = nn.Linear(300, action_size)
        self.mean_layer.weight.data.uniform_(-init_w, init_w)
        self.mean_layer.bias.data.uniform_(-init_w, init_w)

        # small init to last layer weights
        self.log_std_layer = nn.Linear(300, action_size)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

        # print(self.mean_layer,self.mean_layer.parameters())

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std_unclamped = self.log_std_layer(x)
        log_std = torch.clamp(log_std_unclamped, self.log_std_min, self.log_std_max)
        # print (mean,log_std)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):  # deterministic = False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal_distribution = Normal(mean, std)
        e = normal_distribution.rsample()
        action = torch.tanh(e)

        log_prob = normal_distribution.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)#action^2在[0，1]区间内加入一个极小保证运算
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class SACAgent:
    def __init__(self, state_size, action_size, alpha, is_auto_alpha=True, q_lr=LR, policy_lr=LR, a_lr=LR,
                 action_prior="uniform"):  # , action_range
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = [ACTION_LOW, ACTION_HIGH]

        self.alpha = alpha
        self.is_auto_alpha = is_auto_alpha
        self._action_prior = action_prior

        # for the train func. also can be upgraded later on to PER (right now we assume it is false)
        self.prioritized_replay = False

        self.q1_network = SoftQNetwork(self.state_size, self.action_size).to(device)
        self.q2_network = SoftQNetwork(self.state_size, self.action_size).to(device)
        self.q1_target = SoftQNetwork(self.state_size, self.action_size).to(device)
        self.q2_target = SoftQNetwork(self.state_size, self.action_size).to(device)
        self.policy_network = PolicyNetwork(self.state_size, self.action_size).to(device)

        copy_params(self.q1_network, self.q1_target)
        copy_params(self.q2_network, self.q2_target)

        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # for the auto-temperature
        if self.is_auto_alpha:
            # self.alpha = 1
            self.target_entropy = -torch.prod(torch.Tensor((action_size,)).to(device)).item()  # -action_size
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a_lr)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, is_action_discrete=False)
        # self.memory = BasicBuffer(BUFFER_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, t=0):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)  ##
        # self.memory.push(state, action, reward, next_state, done)
        # print(action,"makinta")
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()  ###
            self.learn(experiences, GAMMA)

        # update step count
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # run state in policy net
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print(state)
        mean, log_std = self.policy_network.forward(state)
        std = log_std.exp()
        # print(state.is_cuda)
        # print(mean,log_std)
        # sample from dist
        normal_distribution = Normal(mean, std)
        e = normal_distribution.sample()
        action = torch.tanh(e)
        action = action.cpu().detach().squeeze(0).numpy()

        # rescale action

        # action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0
        # return action
        # rint (action)
        a = action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
            (self.action_range[1] + self.action_range[0]) / 2.0
        # rint (a)
        return a

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards*0.5  #reward scale
        # states = torch.FloatTensor(states).to(device)
        # actions = torch.FloatTensor(actions).to(device)
        # rewards = torch.FloatTensor(rewards).to(device)
        # next_states = torch.FloatTensor(next_states).to(device)
        # dones = torch.FloatTensor(dones).to(device)
        # dones = dones.view(dones.size(0),-1) ###
        # print(states, actions, rewards, next_states, dones)
        # print (states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)

        next_actions, next_log = self.policy_network.sample(next_states)

        Q1_target_next = self.q1_target(next_states, next_actions)
        Q2_target_next = self.q2_target(next_states, next_actions)
        Q_targets_next = torch.min(Q1_target_next, Q2_target_next) - self.alpha * next_log

        Q_targets = rewards + gamma * (Q_targets_next) * (1 - dones)

        Q_1 = self.q1_network.forward(states, actions)
        Q_2 = self.q2_network.forward(states, actions)

        # calc q-nets loss
        Q1_loss = F.mse_loss(Q_1, Q_targets.detach())
        Q2_loss = F.mse_loss(Q_2, Q_targets.detach())

        # update q-nets params
        self.q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.q2_optimizer.step()

        actions_pred, log_pis = self.policy_network.sample(states)

        # Learn every UPDATE_EVERY time steps.
        # IS THIS THE RIGHT ORDER
        if self.t_step == 0:
            Q_min = torch.min(self.q1_network.forward(states, actions_pred),
                              self.q2_network.forward(states, actions_pred))

            policy_loss = (self.alpha * log_pis - Q_min).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.q1_network, self.q1_target, TAU)
            self.soft_update(self.q2_network, self.q2_target, TAU)

        if self.is_auto_alpha:
            alpha_loss = (self.log_alpha * (- log_pis - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)

    def train(self,i,n_episodes=10000, max_t=1300):
        scores = []  # list containing scores from each episode
        avg_scores = []  # list contating avg scores
        scores_window = deque(maxlen=100)  # last 100 scores
        count = 0 #任务成功的次数
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            score = 0
            ok = False
            for t in range(max_t):
                action = self.act(state)
                # do action in environment
                next_state, reward, done, _ = env.step(action)
                ok = env.already_landing
                self.step(state, action, reward, next_state, done)
                # accumulate score and move to next state
                state = next_state
                score += reward
                # stop episode if done
                if done:
                    break
            reporter.add_scalar("reward", score, i_episode)
            reporter.add_scalar("avgstep", env.stepid, i_episode)
            show_state = env.state_buffer
            if ok:
                count+=1
                print('successd')
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            avg_scores.append(np.mean(scores_window))  # save current avg score

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                # uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, i_episode, score)
            if count>1000:
                torch.save(self.policy_network.state_dict(), "SAC_" + str(i) + ".pt")

        return scores, avg_scores

def train(agent, n_episodes=10000, max_t=1250, eps_start=1.0, eps_end=0.01, eps_decay=0.995, agent_type="DQN"):
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
        counts = 0
        for t in range(max_t):
            # choose an action
            if agent_type == "SAC":
                action = agent.act(state)
            # do action in environment
            next_state, reward, done, _ = env.step(action)
            ok = env.already_landing

            # observe and learn (by the agent)
            if agent_type == "SAC":
                agent.step(state, action, reward, next_state, done)
            # accumulate score and move to next state
            state = next_state
            score += reward

            # stop episode if done
            if done:
                break
        reporter.add_scalar("reward",score,i_episode)
        reporter.add_scalar("avgstep", env.stepid, i_episode)
        show_state = env.state_buffer
        if ok:
            uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, i_episode, score)
            print('successd')
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        avg_scores.append(np.mean(scores_window))  # save current avg score


        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, i_episode, score)
        if np.mean(scores_window) >= 9000.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))

            # torch.save(agent.state_dict(), '/saved')


    return scores, avg_scores

def main():
    TOTAL_TIMESTEPS = 5000  # max timesteps, this should be high enough so convergence happens.
    RUNS = 1  # how many runs of each agent (with different seed)
    BETA = LinearSchedule(TOTAL_TIMESTEPS, 1.0, INITIAL_BETA)  # beta schedule for prioritzed replay
    BATCH_SIZE = 64
    LR = 5e-4
    ALPHA_SAC = 0.2
    TAU = 0.01
    UPDATE_EVERY = 2
    # BUFFER_SIZE = 100000#int(1e6)
    sac_scores = []
    sac_avg_scores = []
    sac_times = []
    for i in range(1, RUNS + 1):
        init_seed(i)
        SAC = SACAgent(STATE_SIZE, ACTION_SIZE, ALPHA_SAC)
        start = time.time()
        score, avg_score = train(SAC,  agent_type="SAC")
        end = time.time()
        sac_scores.append(score)
        sac_avg_scores.append(avg_score)
        sac_times.append(end - start)
        torch.save(SAC.policy_network.state_dict(), "SAC_" + str(i) + ".pt")

if __name__ == '__main__':
    main()