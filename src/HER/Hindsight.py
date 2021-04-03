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

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "terminal"])





def hindsight(args):
    """
    Define the hindsight function to be run by a mp process
    The function defines multiple environments which 

    Parameters
    ==========
    args: dictionary containing the funtion configuration parameters. Expected keys:

        "num_actions_per_qubit": actions each qubit can take 
        "device": the computer device which the training will be on (e.g. cpu or gpu)


        "epsilon": (float), probability to choose random action
        "decay_factor_epsilon": how strongly the exploration factor epsilon should decay over time during a training run
        "min_value_factor_epsilon": minimum value that the exploration factor epsilon should be annealed to
        "model_name": (str), specifier for the model
        "model_config": (dict), configuration for the network architecture
            changes with the architecture
        "benchmarking": whether certain performance time measuements should be performed
        "summary_path": (str), base path for tensorboard
        "summary_date": (str), target path for tensorboard of the current run
        "load_model": toggle whether to load a pretrained model
        "old_model_path" if 'load_model' is activated, this is the location from which the old model is loaded
        "discount_factor": gamma factor in reinforcment learning
        "discount_intermediate_reward": the discount factor dictating how strongly lower layers
            should be discounted when calculating the reward for creating/destroying syndromes
        "min_value_factor_intermediate_reward": minmum value that the effect of the intermediate reward should be annealed to
        "decay_factor_intermediate_reward": how strongly the intermediate reward should decay over time during a trining run 
        


    """

    device = args["device"]

    benchmarking = args["benchmarking"]
    num_actions_per_qubit = args["num_actions_per_qubit"]
    epsilon = args["epsilon"]
    load_model_flag = args["load_model"]
    old_model_path = args["old_model_path"]
    discount_factor = args["discount_factor"]
    discount_intermediate_reward = float(args.get("discount_intermediate_reward", 0.75))
    min_value_factor_intermediate_reward = float(args.get("min_value:factor_intermediate_reward", 0.0))