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

def hindsight3(self, config):
    """
    Deep Q-learning with hindsight experience replay
    Create agent network for the hindsight learning. 
    Uses a config dictionary which holds keys bound 
    to neccessary parameters. The neccessary ones are:

    Parameters
    ==========
    n_episodes: (int) maximum number of training episodes
    size_action_history: (int) maximum size of the action history of the environment,
            trying to execute more actions than this in one environment causes the environment
            to terminate and start again with a new syndrome.
    size_local_memory_buffer: local memory buffer size
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
    load_model: (int) 1 or 0, toggle on whether to download a pretrained model or not
    old_model_path: (str) if load_model is 1, then load from this path
    

    model_name: (str) dqn network type
    model_config: (dict) configuration dictionary with neccessary parameters for the neural network
    
    """


    writer = SummaryWriter("runs/" + "BF_HER_4_")

    hindsight_id = config["id"]
    n_episodes = int(config.get("n_episodes"))
    size_action_history = int(config.get("size_action_history"))
    num_actions_per_qubit = int(config.get("num_actions_per_qubit"))
    verbosity = int(config.get("verbosity"))
    batch_size = int(config.get("verbosity"))
    buffer_size = int(config.get("buffer_size"))
    learning_rate = float(config.get("learning_rate"))
    tau = float(config.get("tau"))
    benchmarking = int(config.get("benchmarking"))
    summary_path = str(config.get("summary_path"))
    summary_date = str(config.get("summary_date"))
    
    discount_intermediate_reward = float(config.get("discount_intermediate_reward"))
    discount_factor = foat(config.get("discount_factor", 0))
    min_value_factor_intermediate_reward = float(config.get("min_value_factor_intermediate_reward",0.0))
    min_value_factor_epsilon = float(config.get("min_value_factor_epsilon"))
    epsilon = float(config.get(epsilon, 1))
    load_model_flag = int(config.get("load_model"))
    if load_model_flag == 1:
        old_model_path = str(config.get("old_model_path"))
    decay_factor_intermediate_reward = float(config.get("decay_factor_intermediate_reward", 0.0))
    decay_factor_epsilon = float(config.get("decay_factor_epsilon", 0.0))
    device = str(config.get("device"))
    hindsight_id = int(config.get("id"))
    update_every = int(config.get("update_every", 4))
    n_step = int(config.get(n_step, 1))


    model_name = str(config.get("model_name"))
    model_config = config.get("model_config")
    num_environments = config.get("num_environments")

    logger = basicConfig(level = logging.INFO)
    if verbosity >= 4:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Fire up all the environments!")

    env = SurfaceCode()
    state_size = env.syndrome_size
    code_size = state_size - 1
    stack_depth = env.stack_depth


    #create a collection of independent environments
    environments = EnvironmentSet(env, num_environments)

    #Set the true goal
    true_goal = np.zeros(stack_depth, code_size, code_size)

    transition_type = np.dtype(
        [
            ("state", (np.uint8, (stack_depth, state_size, state_size))),
            ("action", (np.uint8, 3)),
            ("reward", float),
            ("next_state", (np.uint8, (stack_depth, state_size, state_size))),
            ("terminal", bool),
        ]
    )

    #initialize all the environments

    states = environments.reset_all()
    steps_per_episode = np.zeros(num_environments)

    #initialize local memory buffers
    size_local_memory_buffer = config.get("size_local_memory_buffer") + 1
    local_buffer_transitions = np.empty(
        (num_environments, size_local_memory_buffer), dtype=transition_type
    )
    local_buffer_actions = np.empty(
        (num_environments, size_local_memory_buffer, 3), dtype=np.uint8
    )
    local_buffer_qvalues = np.empty(
        (num_environments, size_local_memory_buffer),
        dtype=(float, num_actions_per_qubit * code_size * code_size + 1),
    )
    local_buffer_rewards = np.empty(
        (num_environments, size_local_memory_buffer), dtype=float
    )
    buffer_idx = 0

    replay_buffer = ReplayBuffer()


    #initalize agent
    model_config = extend_model_config(
        model_config, state_size, stack_depth, device=device
    )
    qnetwork_target = choose_model(model_name, model_config)
    qnetwork_local = choose_model(model_name, model_config)

    # load communication queues
    #actor_io_queue = args["actor_io_queue"]
    #learner_actor_queue = args["learner_actor_queue"]


    if load_model_flag == 1:
        qnetwork_target, _, _ = load_model(qnetwork_target, old_model_path)
        qnetwork_local, _, _ = load_model(qnetwork_local, old_model_path)

    qnetwork_target.to(device)
    qnetwork_local.to(device)


    performance_start = time()
    heart = time()
    heartbeat_interval = 60 #seconds

    logger.info(f"Hindsight {hindsight_id} starting to loop in {device}")
    #sent_data_chunks = 0

    #initialize tensorboard for monitoring/logging
    tensorboard = SummaryWriter(os.path.join(summary_path, str(code_size), summary_date, "hindsight"))
    tensorboard_step = 0
    steps_to_benchmark = 0
    benchmark_frequency = 1000

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

        actions, q_values = select_actions(_states, qnetwork_target, code_size, epsilon = annealed_epsilon
        
        if benchmarking and steps_to_benchmark % benchmark_frequency == 0:
            select_action_stop = time()
            logger.info(f"time for select action: {select_action_stop-select_action_start}")

        if verbosity >=2:
            tensorboard.add_scalars("hindsight/epsilon", {"annealed_epsilon": annealed_epsilon}, delta_t, walltime = current_time_tb)


        steps_start = time()

        annealing_intermediate_reward = anneal_factor(delta_t, 
        decay_factor = decay_factor_intermediate_reward,
        min_value = min_value_factor_intermediate_reward)