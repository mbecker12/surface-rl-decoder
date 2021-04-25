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
import DQN_Agent

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "terminal"])


def hindsight(config):
    """
    Deep Q-learning


    Parameters
    ==========
    n_episodes: (int) maximum number of training episodes
    max_t: (int) maximum number of training episodes
    eps_start: (float) starting value of epsilon
    eps_end: (float) minimum value of epsilon
    eps_decay: (float) multiplicative factor (per episode) for decreasing epsilon

    """


    
    writer = SummaryWriter("runs/" + "BF_HER_4_")

    
    frames = int(config.get("frames"))
    eps_fixed = bool(config.get("eps_fixed"))
    eps_frames = int(config.get("eps_frames"))
    min_eps = float(config.get("min_eps"))
    eps_start = config.get("eps_start", 0.999)
    eps_end = config.get("eps_end", 0.001)
    eps_decay = config.get("eps_decay", 0.998)
    env = config.get("env")
    num_environments = config.get("num_environments")
    environments = EnvironmentSet(env, num_environments)

    if eps_fixed:
        eps = 0
    else:
        eps = eps_start
    
    i_episode = 1
    
    transition_type = np.dtype(
        [
            ("state", (np.uint8, (stack_depth, state_size, state_size))),
            ("action", (np.uint8, 3)),
            ("reward", float),
            ("next_state", (np.uint8, (stack_depth, state_size, state_size))),
            ("terminal", bool),
        ]
    )

    #attempt to initialize the set here
    states = environments.reset_all()
    steps_per_episode = np.zeros(num_environments)

    goal_states = np.zeros(stack_depth, state_size, state_size)

    size_local_memory_buffer = config.get("size_local_memory_buffer") + 1
    local_buffer_transitions = np.empty((num_environments, size_local_memory_buffer), dtype = transition_type)
    local_buffer_actions = np.empty((num_environments, size_local_memory_buffer, 3), dtype = np.uint8)
    local_buffer_qvalues = np.empty((num_environments, size_of_local_memory_buffer), (num_actions_per_qubit * code_size * code_size + 1))
    local_buffer_rewards = np.empty((num_environments, size_local_memory_buffer), dtype = float)

    buffer_idx = 0

    agent = DQN_Agent(config)

    scores = np.zeros(num_environments, 1)
    scores_list = []
    scores_window = deque(maxlen = 100)


    for frame in range(1, frames+1): 

        #take steps and states until we are done with the episode 
        while True:

            actions = agent.act(states, eps)
            next_states, rewards, terminals, _ = environments.step(actions)   #Does our environment support this? in becker's code it uses more input variables

            agent.step(states, actions, rewards, next_states, terminals, writer, goal_states) # Check what it produces
            states = next_states
            scores += rewards

            if terminal: #how will this work in the multiple environment? loop through them all using enumerate?
                scores_window.append(scores) #change as well for the sake of the score setting
                scores_list.append(scores)
                writer.add_scalar("Epsilon", eps, i_episode)
                writer.add_scalar("Rewards", scores, i_episode)
                writer.add_scalar("Average100", np.mean(scores_window), i_episode) #not sure if this is correct
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end = "")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)))
                i_episode += 1
                states = environments.reset()
                scores = np.zeros(num_environments, 1)
                break # this may prove a problem as it would create a new episode each time an agent reached this

        if eps_fixed == False:
                if frame < eps_frames:
                    eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
                else:
                    eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
        
    return np.mean(scores_window)




if __name__ == "__main__":

    

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
    env = SurfaceCode()


    verbosity = 4
    action_size = env.action_space.n #check if these are the old one belonging to the original bitflip problem 
    state_size = env.observation_space["state"].shape


    eps_fixed = False

    
    FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = "Her_Log.log", level = logging.DEBUG, format = FORMAT, filemode = "w")
    logger = logging.getLogger(f"HER_{actor_id}")

    config = {"code_size": env.syndrome_size-1,
    "state_size": env.syndrome_size,
    "stack_depth": env.stack_depth,
    "min_qbit_err": 2,
    "p_error": 0.1,
    "p_msmt": 0.05,
    "stack_depth": 8,
    "num_actions_per_qubit": 3,
    "eps_from": 0.999,
    "eps_to": 0.02,
    "eps_decay": 0.998,
    "max_actions": 32,
    "input_channels": 1,
    "kernel_size": 3,
    "output_channels": 20,
    "output_channels2": 50,
    "output_channels3": 30,
    "padding_size": 1,
    "lstm_layerrs": 3,
    "gamma": gamma,
    "lr": lr,
    "tau": tau,
    "model_name": "Conv_2d_agent",
    "bit_length": int(bit_length),
    "eps_fixed": bool(eps_fixed);
    "frames": 35000, 
    "eps_frames": 8000, 
    "min_eps": 0.025,
    "device": str(device),
    "logger": logger,
    "env": env,
    "num_environments": num_environments,
    "actor_id": id,
    "size_action_history": size_action_history,
    "benchmarking": benchmarking,
    "load_model_flag": load_model_flag,
    "old_model_path": old_model_path,
    "verbosity": verbosity
     }


    if verbosity >= 4:
        logger.setLevel(loggin.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Fire up all the environments")

    t0 = time.time()
    final_average100 = hindsight(config)
    t1 = time.time()

    print("Training time: {}min".format(round((t1-t0)/60, 2)))
    torch.save(agent.qnetwork_local.state_dict(), "BF_DQN_HER" + ".pth")