"""

Define hindsight learning process for exploration of environment in
reinforcment learning.¨

"""

import os
from time import time
from collections import deque, namedtuple
import gym
import logging
import random
import math
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import vector_to_parameters, clip_grad_norm
import torch.optim as optim
from distributed.environment_set import EnvironmentSet
from distributed.model_util import choose_model, extend_model_config, load_model
from distributed.util import anneal_factor, compute_priorities, select_actions, time_tb
from surface_rl_decoder.surface_code import SurfaceCode
import ReplayBuffer




def hindsight(config):
    """
    Deep Q-learning


    Parameters
    ==========
    n_episodes: (int) maximum number of training episodes
    max_t: (int) maximum number of training episodes
    eps_start: (float) starting value of epsilon
    eps_end: (float) minimum value of epsilon
    eps_decay: (float) multiplicative factor (per episode) for decreasing epsilom
    """

    #,  frames = 1000, eps_fixed = False, eps_frames = 1e6, min_eps = 0.01
    frames = bool(config.get("frames"))
    eps_fixed = bool(config.get("eps_fixed"))
    eps_frames = int(config.get("eps_frames"))
    min_eps = float(config.get("min_eps"))

    scores = []
    scores_window = deque(maxlen = 100)
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    
    eps_start = 1
    i_episode = 1
    dic = env.reset()
    state = dic["state"]
    goal_state = dic["goal"]
    state = np.concatenate((state, goal_state))
    score = 0

    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        SandG, reward, done, _ = env.step(action)
        next_state = SandG["state"]
        next_state = np.concatenate((next_state, goal_state))

        agent.step(state,action,reward, next_state, done, writer, SandG["goal"])
        state = next_state
        score += reward


        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps -min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
        
        if done:
            scores_window.append(score)
            scores.append(score)
            writer.add_scalar("Epsilon", eps, i_episode)
            writer.add_scalar("Reward", score, i_episode)
            writer.add_scalar("Average100", np.mean(scores_window), i_episode)
            print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end = "")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)))
            i_episode += 1
            dic = env.reset()
            state = dic["state"]
            goal_state = dic["goal"]
            state = np.concatenate((state, goal_state))
            score = 0

    return np.mean(scores_window)




if __name__ == "__main__":
    writer = SummaryWriter("runs/" + "BF_HER_4_")

    FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = "Her_Log.log", level = logging.DEBUG, format = FORMAT, filemode = "w")
    logger = logging.getLogger()

    seed = 3
    buffer_size = 100000
    batch_size = 128
    gamma = 0.98
    tau = 1e-2
    lr = 1e-3
    update_every = 1
    n_stepdevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    bit_length = 16
    np.random.seed(seed)
    env = gym.make("")
    env.__init__(bit_length = bit_length)

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space["state"].shape

    agent = DQN_Agent(config)


    eps_fixed = False

     config = {"code_size": 5,
    "min_qbit_err": 2,
    "p_error": 0.1,
    "p_msmt": 0.05,
    "stack_depth": 8,
    "nr_actions_per_qubit": 3,
    "epsilon_from": 0.999,
    "epsilon_to": 0.02,
    "epsilon_decay": 0.998,
    "max_actions": 32,
    "input_channels": 1,
    "kernel_size": 3,
    "output_channels": 20,
    "output_channels2": 50,
    "output_channels3": 30,
    "padding_size": 1,
    "lstm_layerrs": 3,
    "bit_length": int(bit_length),
    "eps_fixed": bool(eps_fixed);
    "frames": 35000, 
    "eps_frames": 8000, 
    "min_eps": 0.025,
    "device": str(device)
     }

    t0 = time.time()
    final_average100 = run()
    t1 = time.time()

    print("Training time: {}min".format(round((t1-t0)/60, 2)))
    torch.save(agent.qnetwork_local.state_dict(), "BF_DQN_HER" + ".pth")