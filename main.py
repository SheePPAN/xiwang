from agents.algorithm.ppo import PPO
from agents.algorithm.sac import SAC
from agents.agent import Agent
from collections import Counter
from discriminators.gail import GAIL
from discriminators.vail import VAIL
from discriminators.airl import AIRL
from discriminators.vairl import VAIRL
from discriminators.eairl import EAIRL
from discriminators.sqil import SQIL
from utils.utils import RunningMeanStd, Dict, make_transition
import matplotlib.pyplot as plt
from configparser import ConfigParser
from argparse import ArgumentParser

import os
import gymnasium as gym
import numpy as np
import torch
os.makedirs('./model_weights', exist_ok=True)
env = gym.make('Acrobot-v1')
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
print("action dim", action_dim)
print("state_dim", state_dim)
opt = ArgumentParser('parameters')
opt.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
opt.add_argument('--render', type=bool, default=False, help="(default: False)")
opt.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
opt.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
opt.add_argument("--discriminator", type=str, default = 'gail', help = 'discriminator training algorithm(default: gail)')
opt.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
opt.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
opt.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: True)')

args = opt.parse_args()
parser = ConfigParser()
parser.read('config.ini')

demonstrations_location_args = Dict(parser, 'demonstrations_location',True)
agent_args = Dict(parser, args.agent)
discriminator_args = Dict(parser, args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

if args.discriminator == 'airl':
    discriminator = AIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'vairl':
    discriminator = VAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'gail':
    discriminator = GAIL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'vail':
    discriminator = VAIL(writer,device,state_dim, action_dim, discriminator_args)
elif args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'sqil':
    discriminator = SQIL(writer, device, state_dim, action_dim, discriminator_args)
else:
    raise NotImplementedError
    
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
elif args.agent == 'sac':
    algorithm = SAC(device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)

if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
state_rms = RunningMeanStd(state_dim)

score_lst = []
discriminator_score_lst = []
score = 0.0
discriminator_score = 0
if agent_args.on_policy == True:
    state_lst = []
    state_ = (env.reset())[0]
    #print(env.action_space)
    #print(state_)#
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    #print(state_)
    return_list = []
    for n_epi in range(args.epochs):
        reward_list = []
        episode_return = 0
        for t in range(agent_args.traj_length):
            if args.render:    
                env.render()
            state_lst.append(state_)
            action, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
            #print(action.shape)
            #print(action)
            action = torch.squeeze(action)
            #print(action.shape)
            #print(action)
            #print(env.step(action.cpu().numpy()))
            #action = torch.flatten(action)
            #print(action.shape)
            next_state_, r, done, _ , info = env.step(action.detach().cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            if  discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                                        log_prob,\
                                        torch.tensor(state).unsqueeze(0).float().to(device),action,\
                                        torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                              torch.tensor(done).view(1,1)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action).item()

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += r
            episode_return += r
            reward_list.append(r)
            discriminator_score += reward
            if done:
                state_ = (env.reset())[0]
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if writer != None:
                    writer.add_scalar("score/real", score, n_epi)
                    writer.add_scalar("score/discriminator", discriminator_score, n_epi)
                score = 0
                return_list.append(episode_return)
                discriminator_score = 0
                break
            else:
                state = next_state
                state_ = next_state_
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi)
        state_rms.update(np.vstack(state_lst))
        state_lst = []
        return_list.append(episode_return)
        if n_epi%args.print_interval==0 and n_epi!=0 and len(score_lst)!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            dist_count = Counter(reward_list)
            print("Reward dist of episode :{}".format(dist_count))
        score_lst = []
        #return_list.append(1)
        #if (n_epi % args.save_interval == 0 )& (n_epi != 0):
            #torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))
    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SQIL on {}'.format('Acrobot-v1'))
    plt.savefig('SQIL reslts')
    plt.close()

else : #off-policy
    for n_epi in range(args.epochs):
        score = 0.0
        discriminator_score = 0.0
        state = (env.reset())[0]
        done = False
        while not done:
            if args.render:    
                env.render()
            action_, log_prob = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action_.cpu().detach().numpy()
            print(action.shape)
            next_state, r, done, _, info = env.step(action)
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                            log_prob,
                            torch.tensor(state).unsqueeze(0).float().to(device),action_,\
                            torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                  torch.tensor(done).unsqueeze(0)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action_).item()

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            state = next_state

            score += r
            discriminator_score += reward
            
            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("score/discriminator", discriminator_score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))