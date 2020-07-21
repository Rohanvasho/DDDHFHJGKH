import os
import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import os
import sys
import pickle
import gym
import time
import numpy as np
import pandas as pd
import seaborn as sb
import random
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_rl.utils.load_utils import load_policy
from stable_baselines import PPO2
from safe_rl import cpo
import tensorflow as tf
#import matplotlib.pylab as plt
import argparse

def evaluate_learned_policy(env_name, checkpointpath):
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevengeNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"

    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })



    env = VecFrameStack(env, 4)


    agent = PPO2Agent(env, env_type, stochastic)  #defaults to stochastic = False (deterministic policy)
    #agent = RandomAgent(env.action_space)

    learning_returns = []
    max = 0
    maxi=0
    for iter in range (10000,30000,500):
        agent.load(checkpointpath + "/" + str(iter))
        print(checkpointpath)
        episode_count = 30
        tot_r=0
        for i in range(episode_count):
            done = False
            traj = []
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            
            while True:
                action = agent.act(ob, r, done)
                #print(action)
                ob, r, done, _ = env.step(action)

                #print(ob.shape)
                steps += 1
                #print(steps)
                acc_reward += r[0]
                if done:
                    print("steps: {}, return: {}".format(steps,acc_reward))
                    break
            tot_r=tot_r + acc_reward   
            learning_returns.append(acc_reward)
        print((tot_r) / 30, "sdfghjkdfbnm,")
        if (((tot_r) / 30) > max):
        
            max = (tot_r) / 30
            maxi=iter
        
        print(max,maxi,iter,"max_value","max_index","iter")

    env.close()
    #tf.reset_default_graph()



    return learning_returns

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--checkpointpath', default='', help='path to checkpoint to run eval on')
    args = parser.parse_args()
    env_name = args.env_name
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    checkpointpath = args.checkpointpath
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns = evaluate_learned_policy(env_name, checkpointpath)
    returns = pd.DataFrame(returns)
    
    
    fig, ax = plt.subplots()
    returns.hist(ax=ax)
    plt.xlabel("Predicted Rewards")
    plt.ylabel("Frequency")
    #tf.print(x)
    fig.savefig("./eval/"+checkpointpath.replace("/","_")+'_ret.png')
    #write returns to file
    f = open("./eval/" + env_name + checkpointpath.replace("/","_") + "_evaluation0_1.txt",'w')
    for r in returns:
        f.write("{}\n".format(r))
    f.close()