import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from statistics import mean
import physics_sim
import uavutils
from tensorboardX import SummaryWriter
import gym
env = gym.make('Pendulum-v1')
print(env.observation_space.shape.prod())