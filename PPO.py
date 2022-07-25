import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from tensorboardX import SummaryWriter
import uavutils
import physics_sim

torch.manual_seed(0)
np.random.seed(0)
# Hyperparameters
learning_rate = 0.0003
gamma = 0.95
lmbda = 0.96
eps_clip = 0.2
K_epoch = 10
rollout_len = 3
buffer_size = 30
minibatch_size = 32


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, 2)
        self.fc_std = nn.Linear(64, 2)
        self.fc_v = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.reporter = SummaryWriter('runs/PPO2/fifth')

    def pi(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                         torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                         torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()
            advantage_lst = []
            advantage = 0.0
            T = len(delta)
            for t in reversed(range(T)):
                advantage = gamma*lmbda*advantage*done_mask[t]+delta[t]
                advantage_lst.append(advantage.item())
            #for delta_t,mask in zip(delta[::-1],done_mask[::-1]):
                #advantage = gamma * lmbda * advantage*mask + delta_t[0]
                #advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s)
                    dist = Normal(mu, std)
                    entropy = dist.entropy()
                    log_prob = dist.log_prob(a).sum(axis=2).squeeze()
                    ratio = torch.exp(log_prob - old_log_prob.squeeze())  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                    loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(self.v(s), td_target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
                    self.reporter.add_scalar('total_loss', loss.mean().item(), global_step=self.optimization_step)


def main():
    env = physics_sim.PhysicsSim()
    model = PPO()
    score = 0.0
    print_interval = 20
    rollout = []
    show_state = None

    for n_epi in range(10000):
        s = env.reset()
        done = False
        reward = 0
        while not done:
            s = uavutils.randomstatemapping(s)
            with torch.no_grad():
                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                a = torch.clamp(a,-1,1)
                log_prob = dist.log_prob(a).sum()
            s_prime, r, done, info = env.step(uavutils.alloaction_act_2(np.array(a)))
            reward += r
            rollout.append((s, np.array(a), r, uavutils.randomstatemapping(s_prime), log_prob.item(), done))
            model.put_data(rollout)
            rollout = []
            s = s_prime
            score += r
            model.train_net()

        show_state = env.state_buffer
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score / print_interval,
                                                                              model.optimization_step))
            print(len(model.data))
            score = 0.0

        if n_epi % 500 == 0 and n_epi != 0:
            uavutils.save_figure((len(show_state) + 1) * 0.02, len(show_state), show_state, n_epi, reward)


if __name__ == '__main__':
    main()
