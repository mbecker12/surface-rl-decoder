"""
Define the learner process in the multi-process
reinforcement learning setup.
"""
import os
from time import time
import traceback
from typing import Dict
import logging
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter
from distributed.evaluate import evaluate
from distributed.learner_util import (
    log_evaluation_data,
    perform_q_learning_step,
    transform_list_dict,
)
from distributed.model_util import (
    choose_model,
    extend_model_config,
    load_model,
    save_model,
)
from distributed.util import time_tb

def learner(args: Dict):
    """
    Start the learner process. Here, the key learning is performed:
    Transitions are sampled from the io_learner_queue in batches,
    then backpropagation on those batches is performed.

    Parameters
    ==========
    args: dictionary containing important configuration values for
        the learner process. The following keys are expected:
            "learner_io_queue": multiprocessing.Queue
            "io_learner_queue": multiprocessing.Queue
            "verbosity": (int) verbosity level
            "benchmarking": whether certain performance time measurements should be performed
            "device": torch.device
            "syndrome_size": (int), usually code_distance + 1
            "stack_depth": (int), number of layers in syndrome stack
            "learning_rate": learning rate for gradient descent
            "target_update_steps": (int), steps after which to update the target
                network's parameters
            "discount_factor": (float), γ factor in reinforcement learning
            "batch_size": (int), batch_size stochastic gradient descent
            "eval_frequency": (int), steps after which to evaluate policy network
            "learner_eval_p_error": (List), list of different levels of p_error
                to be used in evaluation
            "learner_eval_p_msmt": (List), list of different levels of p_msmt
                to be used in evaluation
            "max_time": (float/int), max learning time in hours
            "max_time_minutes": (float/int), max learning time in minutes
            "timesteps": (int), maximum time steps; set to -1 for infinite time steps
            "model_name": (str) specifier for the model
            "model_config": (dict) configuration for network architecture.
                May change with different architectures
            "learner_epsilon": (float) the exploration probability for the evaluation
                policy
            "summary_path": (str), base path for tensorboard
            "summary_date": (str), target path for tensorboard for current run
            "learner_actor_queues": list of mp.Queues,
                each queue in the list communicates with each of the actors
            "load_model": toggle whether to load a pretrained model
            "old_model_path" if 'load_model' is activated, this is the location from which
                the old model is loaded
            "save_model_path": path to save model & optimizer state_dict and metadata
    """

    # configuration
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    learner_actor_queues = args["learner_actor_queues"]
    verbosity = args["verbosity"]
    benchmarking = args["benchmarking"]
    load_model_flag = args["load_model"]
    old_model_path = args["old_model_path"]
    save_model_path = args["save_model_path"]
    model_name = args["model_name"]
    model_config = args["model_config"]

    learning_rate = args["learning_rate"]
    device = args["device"]
    syndrome_size = args["syndrome_size"]
    code_size = syndrome_size - 1
    stack_depth = args["stack_depth"]
    target_update_steps = args["target_update_steps"]
    discount_factor = args["discount_factor"]
    batch_size = args["batch_size"]
    eval_frequency = args["eval_frequency"]
    p_error_list = args["learner_eval_p_error"]
    p_msmt_list = args["learner_eval_p_msmt"]
    learner_epsilon = args["learner_epsilon"]
    count_to_eval = 0

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("learner")
    if verbosity >= 4:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]
    save_model_path_date = os.path.join(
        save_model_path,
        str(code_size),
        summary_date,
        f"{model_name}_{code_size}_{summary_date}.pt",
    )

    start_time = time()
    max_time_h = args["max_time"]  # hours
    max_time_min = float(args.get("max_time_minutes", 0))  # minutes
    max_time = max_time_h * 60 * 60  # seconds
    max_time += max_time_min * 60  # seconds

    heart = time()
    heartbeat_interval = 60  # seconds
    timesteps = args["timesteps"]
    if timesteps == -1:
        timesteps = np.Infinity

    # initialize models and other learning gadgets
    model_config = extend_model_config(
        model_config, syndrome_size, stack_depth, device=device
    )

    policy_net = choose_model(model_name, model_config)
    target_net = choose_model(model_name, model_config)

    

    if load_model_flag:
        policy_net, optimizer, criterion = load_model(
            policy_net,
            old_model_path,
            load_optimizer=True,
            load_criterion=True,
            optimizer_device=device,
            model_device=device,
            learning_rate=learning_rate,
        )
        target_net, _, _ = load_model(target_net, old_model_path, model_device=device)
        logger.info(f"Loaded learner models from {old_model_path}")
    else:
        policy_net.to(device)
        target_net.to(device)
        criterion = nn.MSELoss(reduction="none")
        optimizer = Adam(policy_net.parameters(), lr=learning_rate)

    # initialize tensorboard
    tensorboard = SummaryWriter(
        os.path.join(summary_path, str(code_size), summary_date, "learner")
    )
    tensorboard_step = 0
    received_data = 0



