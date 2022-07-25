import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import physics_sim
import uavutils

torch.manual_seed(0)
np.random.seed(0)
# Hyperparameters
lr_mu = 0.0003  # actor
lr_q = 0.003  # critic
gamma = 0.97
batch_size = 32
buffer_limit = 100000
tau = 0.005  # for target network soft update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

    def pop(self):
        self.buffer.popleft()


class MuNet(nn.Module):  # Actor
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc_mu = nn.Linear(64, 2)  # 5个动作两个隐藏层

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))  # 限制在[-1,1]
        return mu

    def initialize(self):  # 将最后一层初始化权重
        nn.init.uniform_(self.fc_mu.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_mu.bias.data, -0.003, 0.003)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(2, 64)
        self.fc_q = nn.Linear(128, 50)
        self.fc_out = nn.Linear(50, 1)

    def forward(self, x, a):
        h1 = torch.tanh(self.fc_s(x))
        h2 = torch.tanh(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.tanh(self.fc_q(cat))
        q = self.fc_out(q)
        return q

    def initialize(self):  # 将最后一层初始化权重
        nn.init.uniform_(self.fc_out.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_out.bias.data, -0.003, 0.003)


class OrnsteinUhlenbeckNoise:  # dt是我们的步长
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.02, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, reporter, timestep):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    reporter.add("q_loss", q_loss, timestep)
    xxx = -q(s, mu(s))
    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    reporter.add("actor_loss", mu_loss, timestep)


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main():
    env = physics_sim.PhysicsSim()
    memory = ReplayBuffer()
    q, q_target = QNet(), QNet()
    q.initialize()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu.initialize()
    mu_target.load_state_dict(mu.state_dict())
    score = 0.0
    avg_step = 0.0
    print_interval = 20
    reporter = uavutils.Recording(name='runs/DDPG/first')
    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q, weight_decay=0.02)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(2))
    trainstep = 0
    for n_epi in range(10000):  # 训练10000回合
        s = env.reset()
        reward = 0
        while not env.done:
            s = uavutils.randomstatemapping(s)
            a = mu(torch.from_numpy(s).float())
            a = a.detach().numpy() + ou_noise()
            a = np.clip(a, -1, 1)
            s_prime, r, done, info = env.step(uavutils.alloaction_act_2(a))
            if memory.size() == buffer_limit:
                memory.pop()
            memory.put((s, a, r / 100, uavutils.randomstatemapping(s_prime), done))  # 对于奖励进行放缩
            score += r
            reward += r
            s = s_prime
        avg_step += env.stepid
        reporter.add("reward", reward, n_epi)
        reporter.add("max_step", env.stepid, n_epi)
        show_state = env.state_buffer
        if memory.size() > 5000:
            for i in range(8):  # 训练8次
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, reporter, trainstep)
                trainstep+=1
                soft_update(mu, mu_target)
                soft_update(q, q_target)
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f} , avg step : {}".format(n_epi, score / print_interval,
                                                                                avg_step / print_interval))
            score = 0.0
            avg_step = 0.0
        if n_epi % 5000 == 0 and n_epi >= 5000 and memory.size() > 5000:
            uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, n_epi, reward)
    torch.save(
        {'epoch': 300000, 'Critic_dict': q.state_dict(), 'Actor_dict': mu.state_dict(),
         'optimizer_a': mu.optimizer_a.state_dict(), 'optimizer_c': q.optimizer_c.state_dict()},
        'last' + '.pth.tar')


if __name__ == '__main__':
    main()
