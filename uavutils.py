import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import gym
import os


class state_normalization(object):
    # Use Welford's algorithm
    def __init__(self):
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

    def normalize_state(self, state):
        state = torch.tensor(state)
        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))

        if len(state.size()) == 1:
            state_old = self.welford_state_mean
            self.welford_state_mean += (state - state_old) / self.welford_state_n
            self.welford_state_mean_diff += (state - state_old) * (state - state_old)
            self.welford_state_n += 1
        else:
            raise RuntimeError
        rt = (state - self.welford_state_mean) / np.sqrt(self.welford_state_mean_diff / self.welford_state_n)
        return rt.numpy()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1 - float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)


class Recording(object):
    def __init__(self, name=None):
        self.reporter = SummaryWriter(name)

    def add(self, name, loss, step):
        self.reporter.add_scalar(name, loss, step)

    def add_weight(self, name, obj, step):
        self.reporter.add_histogram(name, values=obj, global_step=step)


def actionmapping(action):  # 从（-1，1）折射成期望动作
    old_action = np.clip(action, -1, 1)
    thrust = np.array([old_action[0] * 0.732 * 1225 + 1225])
    actutator = old_action[1:] * 0.3491
    return np.concatenate((thrust, actutator))


def alloaction_act(action):
    u_limit = 0.3491
    pinv = np.array([[-0.9273, 0, 1.1912],
                     [0, -0.9273, 1.1912],
                     [0.9273, 0, 1.1912],
                     [0, 0.9273, 1.1912]])
    thrust = action[0] * 0.732 * 1225 + 1225
    f = np.clip(action[1:], -1, 1) * np.array([0.2, 0.2, 0.44])
    u = np.matmul(pinv, f)
    return np.concatenate((np.array([thrust]), u))


# 只有两个动作版本
def alloaction_act_2(action):
    u_limit = 0.3491
    action = np.array([action[0], 0, 0, action[1]])
    action = np.clip(action, -1, 1)
    pinv = np.array([[-0.9273, 0, 1.1912],
                     [0, -0.9273, 1.1912],
                     [0.9273, 0, 1.1912],
                     [0, 0.9273, 1.1912]])

    thrust = action[0] * 0.732 * 1225 + 1225
    f = np.clip(action[1:], -1, 1) * np.array([1.13, 1.13, 0.44])
    u = np.matmul(pinv, f)
    return np.concatenate((np.array([thrust]), u))


def randomactionmapping(old_action):
    thrust = np.array([old_action[0] * 0.732 * 1225 + 1225])
    actutator = old_action[1:] * np.pi
    return np.concatenate((thrust, actutator))


def statemapping(state):  # 归一到（0，1）
    time = state[-1]
    x = state[-4]
    y = state[-3]
    z = state[-2]
    V = state[6:9] / 10
    ang_v = state[9:12] / (2 * np.pi)
    pos = np.array([x / 5, y / 5, z / 15 - 1,time])

    return np.concatenate((state[0:6], V, ang_v, pos))


def randomstatemapping(state):  # 归一到（0，1）
    z, Vz, r = state

    return np.array([z / 12.5 - 1, Vz / 10, r / np.pi])


def save_figure(total_time, stepid, state_buffer, episode, reward):
    label = ['Ub', 'Vb', 'Wb', 'p', 'q', 'r', 'Xe', 'Ye', 'Ze', 'PITCH', 'ROLL', 'YAW', 'Vx', 'Vy', 'Vz']
    t = np.linspace(0, total_time + 0.02, stepid)
    data = list(map(list, zip(*state_buffer)))
    fig, ax = plt.subplots(5, 3)
    for i in range(15):
        plt.subplot(5, 3, i + 1)
        plt.plot(t, data[i])
        plt.title(label[i])
    plt.tight_layout()
    plt.savefig('test1/{}episode_reward_{}.jpg'.format(episode, reward), bbox_inches='tight')


def save_figure_reward(total_time, stepid, state_buffer, episode, reward):
    label = ['thrust', 'c1', 'c2', 'c3', 'c4']
    t = np.linspace(0, total_time + 0.02, stepid)
    data = list(map(list, zip(*state_buffer)))
    fig, ax = plt.subplots(4, 1)
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(t, data[i])
        plt.title(label[i])
    plt.tight_layout()
    plt.savefig('test1/{}episode_reward_action{}.jpg'.format(episode, reward), bbox_inches='tight')
    plt.close()


def save_action(action_buffer, n_epi):
    label = ['thrust']
    t = np.arange(len(action_buffer))
    plt.plot(t, action_buffer)
    plt.title(label)
    plt.savefig('test1/{}episode_action.jpg'.format(n_epi), bbox_inches='tight')
    plt.close()
