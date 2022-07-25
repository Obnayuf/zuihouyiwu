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

torch.manual_seed(0)
np.random.seed(0)
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_', 'done'])
gamma = 0.99
beta = 0.001
lmbda = 0.95


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.fc1 = nn.Linear(128, 64)
        self.mu_head = nn.Linear(64, 1)
        self.sigma_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.fc1 = nn.Linear(128, 64)
        self.v_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        state_value = self.v_head(x)
        return state_value


class PPO():
    clip_param = 0.1
    max_grad_norm = 0.5
    ppo_epoch = 3
    buffer_capacity, batch_size = 1024, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0
        self.reporter = SummaryWriter('runs/PPO2/third')
        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        (mu, sigma) = self.anet(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def calc_advantage(self, s, a, r, s_, done):
        gaes = torch.zeros_like(r)
        future_gae = torch.tensor(0.0, dtype=r.dtype)
        with torch.no_grad():  # 使用squeeze降维
            deltas = r + gamma * self.cnet(s_).squeeze() * (1 - done) - self.cnet(s).squeeze()  # 去计算此时的T时间内每一个状态的delta（GAE）
            T = len(deltas)
            for t in reversed(range(T)):
                gaes[t] = future_gae = deltas[t] + gamma * lmbda * (1 - done[t]) * future_gae  # 迭代更新
                target_v = gaes + self.cnet(s).squeeze()  # P111
        return gaes, target_v

    def update(self):
        self.training_step += 1
        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)
        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float)

        done = torch.tensor([t.done for t in self.buffer], dtype=torch.int)
        gaes, target_v = self.calc_advantage(s, a, r, s_, done)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size,
                    False):  # 从容量1000中随机抽取32个step去训练，一共32次，32*32=1024

                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu.squeeze(), sigma.squeeze())
                entropy = dist.entropy()  # 哪一维度希望为1哪一个维度是1
                action_log_prob = dist.log_prob(a[index])
                ratio = torch.exp(action_log_prob - (old_action_log_probs[index]))
                surr1 = ratio * gaes[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * gaes[index]
                action_loss = -torch.min(surr1, surr2)  # index是一个1*32的数组，因此我们每从1000个步长里抽取32个，进行一次更新，更新32次
                total_loss = action_loss - beta * entropy
                self.optimizer_a.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)  # 防止梯度爆炸
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]).squeeze(), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()
                self.reporter.add_scalar('value_loss', value_loss.mean().item(), global_step=self.training_step)
                self.reporter.add_scalar('actor_loss', action_loss.mean().item(), global_step=self.training_step)
                self.reporter.add_scalar('entropy', entropy.mean().item(), global_step=self.training_step)
                self.reporter.add_scalar('total_loss', total_loss.mean().item(), global_step=self.training_step)
        del self.buffer[:]  # 清空buffer中的数据


def main():
    #env = physics_sim.PhysicsSim()
    env = gym.make('Pendulum-v1')
    agent = PPO()
    score = 0
    show_state = None
    prev_reward = None
    for i_ep in range(10000):
        state = env.reset()
        done = False
        standard_reward = 0
        while not done:
            action, action_log_prob = agent.select_action(state)
            state_, reward, done, _ = env.step([action*2])
            if agent.store(Transition(state, action, action_log_prob, reward,
                                      state_, done)):
                agent.update()
            score += reward
            standard_reward += reward
            state = state_
        # agent.reporter.add_scalar('reward', standard_reward, i_ep)
        # agent.reporter.add_scalar('max_step', env.stepid, i_ep)
        # show_state = env.state_buffer
        # if env.already_landing:
        #     torch.save(
        #         {'epoch': i_ep + 1, 'Critic_dict': agent.cnet.state_dict(), 'Actor_dict': agent.anet.state_dict(),
        #          'optimizer_a': agent.optimizer_a.state_dict(), 'optimizer_c': agent.optimizer_c.state_dict()},
        #         'EP{}reward{}'.format(i_ep, standard_reward) + '.pth.tar')

        if i_ep % 20 == 0 and i_ep!=0:
            print('Ep {}\t average score: {:.2f}\t'.format(i_ep, score / 20))
            # uavutils.save_figure(env.time, env.stepid, env.statebuffer, i_ep, standard_reward)
            score = 0.0
    #     if i_ep % 500 == 0 and i_ep!=0:
    #         uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, i_ep, prev_reward)
    #     if i_ep % 1000 ==0 and i_ep!=0:
    #         print(agent.anet.state_dict())
    # torch.save(
    #         {'epoch': 9999 + 1, 'Critic_dict': agent.cnet.state_dict(), 'Actor_dict': agent.anet.state_dict(),
    #          'optimizer_a': agent.optimizer_a.state_dict(), 'optimizer_c': agent.optimizer_c.state_dict()},
    #         'EP{}reward{}'.format(9999, 9999) + '.pth.tar')


if __name__ == '__main__':
    main()
