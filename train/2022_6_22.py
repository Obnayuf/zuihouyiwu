import torch
from algorithm import truesac
from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
from physics_sim import PhysicsSim
import numpy as np

def evaluate_policy(env, agent):
    times = 100
    evaluate_reward = 0
    success_counts = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.act(s,deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if env.already_landing:
                success_counts += 1
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times,success_counts/times



def main(args, env_name , time , seed):
    writer = SummaryWriter(log_dir='runs/SAC/env_{}__time_{}_seed_{}'.format(env_name,time, seed))
    truesac.init_seed(seed)
    env = PhysicsSim()
    env_evaluate = PhysicsSim()
    env.set_seed(seed)
    env_evaluate.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = truesac.SACAgent(state_size=16,action_size=4,alpha=args.alpha,batch_size=args.mini_batch_size,
                             buffer_size=args.buffer_size,gamma=args.gamma,reward_scale=args.reward_scale,
                             policy_units_1=args.hidden_width_1 ,value_units_1=args.hidden_width_1,
                             policy_units_2=args.hidden_width_2,value_units_2=args.hidden_width_2,reporter=writer)
    evaluate_num = 0
    evaluate_rewards = []
    for i_episode in range(1, args.max_episodes + 1):
        state = env.reset()
        score = 0
        counts = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            # choose an action
            action = agent.act(state)
            # do action in environment
            next_state, reward, done, _ = env.step(action)
            # observe and learn (by the agent)
            if done and episode_steps != 1250:
                dw = True
            else:
                dw = False
            agent.step(state, action, reward, next_state, dw)
            # accumulate score and move to next state
            state = next_state
            score += reward

        if i_episode%500 == 0:
            evaluate_num += 1
            evaluate_reward,success_rate = evaluate_policy(env_evaluate,agent)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t 成功率：{} \t".format(evaluate_num, evaluate_reward,
                                                                              success_rate))
            writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=i_episode)







if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for soft actor-critic")
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy of policy")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--tau", type=float, default=0.01, help="soft update rate")
    parser.add_argument("--reward_scale", type=float, default=5, help="reward scale para")
    parser.add_argument("--hidden_width_1", type=int, default=400,
                        help="The number of neurons in first hidden layers of the neural network")
    parser.add_argument("--hidden_width_2", type=int, default=300,
                        help="The number of neurons in second hidden layers of the neural network")
    parser.add_argument("--max_episodes" , type=int,default=10000)
    args = parser.parse_args()
    env_name = "ducted_fan"
    TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
    main(args,env_name,TIMESTAMP,seed=10)
