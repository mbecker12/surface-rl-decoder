"""

Define hindsight learning process for exploration of environment in
reinforcment learning.¨

"""

import os
from time import time
from collections import namedtuple
#import gym
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
from copy import deepcopy
from distributed.environment_set import EnvironmentSet
from distributed.model_util import choose_model, extend_model_config, load_model, save_model
from distributed.util import anneal_factor, select_actions, time_tb
from surface_rl_decoder.surface_code import SurfaceCode
from replay_buffer import ReplayBuffer


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "terminal", "goal"])

def hindsight(config):
    """
    Deep Q-learning with hindsight experience replay
    Create agent network for the hindsight learning. 
    Uses a config dictionary which holds keys bound 
    to neccessary parameters. The neccessary ones are:

    Parameters
    ==========
    n_episodes: (int) maximum number of training episodes
    epoch_steps: (int) number of steps in an epoch before it goes over to the training section
    size_action_history: (int) maximum size of the action history of the environment,
            trying to execute more actions than this in one environment causes the environment
            to terminate and start again with a new syndrome.
    num_actions_per_qubit: (int) number of possible operators in a qubit
    verbosity: (int) verbosity level
    batch_size: (int) size of batches
    buffer_size: (int) size of replay memory
    learning_rate: (float) learning rate of the algorithm
    tau: (float) tau for soft updating the network weights
    benchmarking: (int) 1 or 0, whether certain performance time measurements should be performed
    summary_path: (str) base path for tensorboard
    summary_date: (str) target path for tensorboard for the current run
    discount_factor: (float) gamma discount factor
    discount_intermediate_reward: (float) the discount factor 
    min_value_factor_epsilon: (float) smallest value of epsilon
    min_value_factor_intermediate_reward: (float) minimum value that 
            the effect of the intermediate reward should be annealed to
    decay_factor_intermediate_reward: (float) how fast the intermediate 
            reward should decay to
    decay_factor_epsilon: how fast should epsilon decay over the training
    device: (str) device that is used for computation
    id: (int) id for process for identification
    update_every: (int) update frequency
    n_step: (int) n_step for how far one should consider the rewards
    n: (int) how many extra goals to have
    load_model: (int) 1 or 0, toggle on whether to download a pretrained model or not
    old_model_path: (str) if load_model is 1, then load from this path
    save_model_path: (str) path to save model & optimizer state_dict and metadata
    

    model_name: (str) dqn network type
    model_config: (dict) configuration dictionary with neccessary parameters for the neural network
    
    """



    hindsight_id = config.get("id")
    #n_episodes = int(config.get("n_episodes"))
    size_action_history = int(config.get("size_action_history"))
    num_actions_per_qubit = int(config.get("num_actions_per_qubit"))
    verbosity = int(config.get("verbosity"))
    batch_size = int(config.get("verbosity"))
    buffer_size = int(config.get("buffer_size"))
    learning_rate = float(config.get("learning_rate"))
    tau = float(config.get("tau"))

    benchmarking = int(config.get("benchmarking"), 0)
    summary_path = str(config.get("summary_path"))
    summary_date = str(config.get("summary_date"))
    
    epsilon = float(config.get("epsilon", 1))
    load_model_flag = int(config.get("load_model"))
    if load_model_flag == 1:
        old_model_path = str(config.get("old_model_path"))

    save_model_path = config.get("save_model_path")

    discount_intermediate_reward = float(config.get("discount_intermediate_reward"))
    discount_factor = float(config.get("discount_factor", 0))
    min_value_factor_intermediate_reward = float(config.get("min_value_factor_intermediate_reward",0.0))
    min_value_factor_epsilon = float(config.get("min_value_factor_epsilon"))    
    decay_factor_intermediate_reward = float(config.get("decay_factor_intermediate_reward", 0.0))
    decay_factor_epsilon = float(config.get("decay_factor_epsilon", 0.0))
    device = str(config.get("device"))
    hindsight_id = int(config.get("id"))
    #update_every = int(config.get("update_every", 4))
    n_step = int(config.get("n_step", 1))
    n = int(config.get("n",4))


    model_name = str(config.get("model_name"))
    model_config = config.get("model_config")
    num_environments = config.get("num_environments")


    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(f"hindsight_{hindsight_id}")
    if verbosity >= 4:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Fire up all the environments!")

    env = SurfaceCode()
    state_size = env.syndrome_size
    code_size = state_size - 1
    stack_depth = env.stack_depth

    
    save_model_path_date = os.path.join(save_model_path, str(code_size), summary_date, f"{model_name}_{code_size}_{summary_date}.pt")

    #create a collection of independent environments
    environments = EnvironmentSet(env, num_environments)

    transition_type = np.dtype(
        [
            ("state", (np.uint8, (stack_depth, state_size, state_size))),
            ("action", (np.uint8, int)),
            ("reward", float),
            ("next_state", (np.uint8, (stack_depth, state_size, state_size))),
            ("terminal", bool),
            ("goal", (np.uint8, (stack_depth, state_size, state_size)))
        ]
    )

    #Set the true goal
    true_goal = np.zeros(stack_depth, state_size, state_size)

    #initialize all the environments
    states = environments.reset_all()
    steps_per_episode = np.zeros(num_environments)

    #initialize local memory buffer
    size_local_memory_buffer = config.get("size_local_memory_buffer") + 1
    local_buffer_transitions = np.empty((num_environments, size_local_memory_buffer), dtype = transition_type)
    local_buffer_qvalues = np.empty((num_environments, size_local_memory_buffer), dtype = (float, num_actions_per_qubit * code_size * code_size + 1))

    buffer_idx = np.zeros(num_environments)
    
    replay_buffer = ReplayBuffer(buffer_size, batch_size, device, discount_factor, n_step)


    #initalize agent
    model_config = extend_model_config(
        model_config, state_size, stack_depth, device=device
    )
    qnetwork_target = choose_model(model_name, model_config)
    qnetwork_local = choose_model(model_name, model_config)

    if load_model_flag == 1:
        qnetwork_target, _, _ = load_model(qnetwork_target, 
        old_model_path,
        model_device = device)

        qnetwork_local, optimizer, criterion = load_model(qnetwork_local, 
        old_model_path, 
        load_optimizer = True, 
        optimizer_device = device, 
        model_device = device, 
        learning_rate = learning_rate)

    else:
        
        qnetwork_target.to(device)
        qnetwork_local.to(device)
        optimizer = optim.Adam(qnetwork_local.parameters(), lr = learning_rate)
        criterion = nn.MSELoss(reduction = "none")



    
    
    performance_start = time()
    heart = time()
    heartbeat_interval = 60 #seconds

    logger.info(f"Hindsight {hindsight_id} starting to loop in {device}")


    #initialize tensorboard for monitoring/logging
    tensorboard = SummaryWriter(os.path.join(summary_path, str(code_size), summary_date, "hindsight"))
    tensorboard_step = 0
    steps_to_benchmark = 0
    benchmark_frequency = 1000

    epoch_steps = int(config.get("epoch_steps", 100000)) 
    steps_in_this_epoch = 0
    Q_updates = 0

    while True:
        steps_per_episode += 1
        steps_to_benchmark += 1

        #select actions based on the chosen model and latest states
        _states = torch.tensor(states, dtype = torch.float32, device = device)
        select_action_start = time()
        current_time_tb = time_tb()
        delta_t = select_action_start - performance_start

        annealed_epsilon = anneal_factor(delta_t, 
        decay_factor = decay_factor_epsilon,
        min_value = min_value_factor_epsilon,
        base_factor = epsilon)

        actions, q_values = select_actions(_states, qnetwork_target, code_size, epsilon = annealed_epsilon)
        
        if benchmarking and steps_to_benchmark % benchmark_frequency == 0:
            select_action_stop = time()
            logger.info(f"time for select action: {select_action_stop-select_action_start}")

        if verbosity >=2:
            tensorboard.add_scalars("hindsight/epsilon", {"annealed_epsilon": annealed_epsilon}, delta_t, walltime = current_time_tb)


        steps_start = time()

        annealing_intermediate_reward = anneal_factor(delta_t, 
        decay_factor = decay_factor_intermediate_reward,
        min_value = min_value_factor_intermediate_reward)

        next_states, rewards, terminals, _ = environments.step(actions, 
        discount_intermediate_reward = discount_intermediate_reward,
        annealing_intermediate_reward = annealing_intermediate_reward,
        punish_repeating_actions = 0)

        if benchmarking and steps_to_benchmark % benchmark_frequency == 0:
            steps_stop = time()
            logger.info(f"time to step through environments: {steps_stop-steps_start}")

        if verbosity >= 2:
            current_time_tb = time_tb()
            tensorboard.add_scalars("hindsight/effect_intermediate_reward",
            {"anneal_factor": annealing_intermediate_reward},
            delta_t,
            walltime = current_time_tb)
        
        
        #send data to the local buffer and if episode complete also to the replay buffer
        transitions = np.asarray(
            [
                Transition(
                    states[i], actions[i], rewards[i], next_states[i], terminals[i], true_goal
                )
                for i in range(num_environments)
            ],
            dtype = transition_type,
        )
        for i in range(num_environments):
            local_buffer_transitions[i, buffer_idx[i]] = transitions[i]
            local_buffer_qvalues[i, buffer_idx[i]] = q_values[i]
        buffer_idx += 1

    
        if verbosity >= 4:
            tensorboard.add_scalar("hindsight/transitions",
            num_environments * steps_in_this_epoch,
            delta_t,
            walltime = current_time_tb)
            tensorboard_step += 1


        #if anywhere has a finished episode or there is too much data then push it to the memory
        if np.any(buffer_idx) >= size_local_memory_buffer or np.any(terminals):
            indices = np.argwhere(np.logical_or(terminals, buffer_idx >= size_local_memory_buffer)).flatten
            for index in indices:
                for step in range(buffer_idx[index]+1):
                    replay_buffer.add(local_buffer_transitions[index,step][0], #state
                    np.argmax(local_buffer_qvalues[index,step]), #action in terms of theneural network
                    local_buffer_transitions[index, step][2], # reward
                    local_buffer_transitions[index,step][3], #next_state
                    local_buffer_transitions[index, step][4],  #terminal
                    local_buffer_transitions[index, step][5]) #goal 
                replay_buffer.new_goals(index, buffer_idx[index], local_buffer_transitions, local_buffer_qvalues, n)
                N = buffer_idx[index]
                for _ in range(N):
                    #If enough samples are available in memory, get random subset and learn
                    if len(replay_buffer) > batch_size:
                        experiences = replay_buffer.sample()
                        #learn(experiences, qnetwork_local, qnetwork_target, optimizer, discount_factor, tau, n_step)
                        loss = learn(experiences, qnetwork_local, qnetwork_target, optimizer, discount_factor, tau, n_step)
                        Q_updates += 1
                        tensorboard.add_scalar("Q_loss", loss, Q_updates)
                buffer_idx[index] = 0
                
            

        too_many_steps = steps_per_episode > size_action_history

        if np.any(terminals) or np.any(too_many_steps):
            #find terminal envs
            indices = np.argwhere(np.logical_or(terminals, too_many_steps)).flatten

            reset_states = environments.reset_terminal_environments(indices = indices)
            next_states[indices] = reset_states[indices]
            steps_per_episode[indices] = 0

            #write to the score window

        states = next_states
        environments.states = deepcopy(states)

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("It's alive")


        steps_in_this_epoch += 1
        if steps_in_this_epoch == epoch_steps:
            break

    save_model(qnetwork_local, optimizer, criterion, save_model_path_date)
    logger.info(f"Saved policy network to {save_model_path_date}")




def learn(experiences, qnetwork_local, qnetwork_target, optimizer, gamma, tau, n_step):
    """Update value parameters of provided models using given batch of experience tuples
    Parameters
    ==========
        experiences: (Tuple[torch.Tensor]) tuple of (s,a,r,s', terminals, goals) tuples
        gamma: (float) discount factor
        tau: interpolation factor for the soft update
        qnetwork_local: model for prediction
        qnetwork_target: model for the target network
    """
    optimizer.zero_grad()
    states, actions, rewards, next_states, terminals, _ = experiences
    #Get max predicted Q values (for next states) from target model
    Q_targets_next = qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    #Compute Q targets for current states
    Q_targets = rewards + (gamma**n_step * Q_targets_next * ( 1 - terminals))
    #Get expected Q values from local model
    Q_expected = qnetwork_local(states).gather(1, actions)
    #Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)
    #Minimise the loss
    loss.backward()
    clip_grad_norm(qnetwork_local.parameters(),1)
    optimizer.step()

    #Update target network
    soft_update(qnetwork_local, qnetwork_target, tau)
    return loss.detach().cpu().numpy()

def soft_update(local_model, target_model, tau):
    """Soft update model parameters
    Parameters
    ==========
        local_model: (Pytorch model) weights will be copied from
        target_model (Pytorch model): weights will be copied to
        tau: (float) interpolation parameter
    """

    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


