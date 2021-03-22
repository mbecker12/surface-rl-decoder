"""
Define the learner process in the multi-process
reinforcement learning setup.
"""
import os
from time import time, sleep
import traceback
from typing import Dict
import logging
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter
from distributed.dummy_agent import DummyModel
from distributed.evaluate import evaluate
from distributed.learner_util import perform_q_learning_step
from model_util import (
    choose_model,
    extend_model_config,
    load_model,
    optimizer_to,
    save_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learner")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-locals, too-many-statements
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
            "device": torch.device
            "syndrome_size": (int), usually code_distance + 1
            "stack_depth": (int), number of layers in syndrome stack
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
            "timesteps": (int), maximum time steps; set to -1 for infinite time steps
            "model_name": (str) specifier for the model
            "model_config": (dict) configuration for network architecture.
                May change with different architectures
            "learner_epsilon": (float) the exploration probability for the evaluation
                policy
            "summary_path": (str), base path for tensorboard
            "summary_date": (str), target path for tensorboard for current run
    """
    # configuration
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    learner_actor_queue = args["learner_actor_queue"]
    verbosity = args["verbosity"]
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

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]
    save_model_path_date = os.path.join(
        save_model_path, summary_date, f"{model_name}_{code_size}_{summary_date}.pt"
    )

    start_time = time()
    max_time_h = args["max_time"]  # hours
    max_time = max_time_h * 60 * 60  # seconds

    heart = time()
    heartbeat_interval = 60  # seconds
    timesteps = args["timesteps"]
    if timesteps == -1:
        timesteps = np.Infinity

    # initialize models and other learning gadgets
    model_config = extend_model_config(model_config, syndrome_size, stack_depth)

    policy_net = choose_model(model_name, model_config)
    target_net = choose_model(model_name, model_config)

    criterion = nn.MSELoss(reduction="none")

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
        optimizer = Adam(policy_net.parameters(), lr=learning_rate)

    # initialize tensorboard
    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "learner"))
    tensorboard_step = 0
    received_data = 0

    # start the actual learning
    t = 0
    while t < timesteps:
        t += 1
        count_to_eval += 1

        if time() - start_time > max_time:
            logger.warning("Learner: time exceeded, aborting...")
            break

        # after a certain number of steps, update the frozen target network
        if t % target_update_steps == 0 and t > 0:
            logger.debug("Update target network parameters")
            params = parameters_to_vector(policy_net.parameters())
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device)

            # notify the actor process that its network parameters should be updated
            msg = ("network_update", params.detach())
            logger.info("Send network weights to actor process")
            learner_actor_queue.put(msg)

        if io_learner_queue.qsize == 0:
            logger.debug("Learner waiting")

        # receive new data from replay memory
        data = io_learner_queue.get()
        if data is not None:
            transitions = data[0]
            data_size = len(transitions)
            received_data += data_size
            assert data_size == batch_size, data_size

            if verbosity:
                tensorboard.add_scalar(
                    "learner/received_data", received_data, tensorboard_step
                )
                tensorboard_step += 1

        # TODO: in this whole nn section,
        # we might need an abstraction layer to support
        # different learning strategies

        # perform the actual learning
        try:
            indices, priorities = perform_q_learning_step(
                policy_net,
                target_net,
                device,
                criterion,
                optimizer,
                data,
                code_size,
                batch_size,
                discount_factor,
                logger=logger,
                verbosity=verbosity,
            )

            # update priorities in replay_memory
            p_update = (indices, priorities)
            msg = ("priorities", p_update)
            learner_io_queue.put(msg)
        except TypeError as _:
            error_traceback = traceback.format_exc()
            logger.error("Caught exception in learning step")
            logger.error(error_traceback)

        # evaluate policy network
        if eval_frequency != -1 and count_to_eval >= eval_frequency:
            logger.info(f"Start Evaluation, Step {t}")
            count_to_eval = 0
            success_rate, ground_state_rate, _, mean_q_list, _ = evaluate(
                policy_net,
                "",
                device,
                p_error_list,
                p_msmt_list,
                plot_one_episode=False,
                epsilon=learner_epsilon,
            )

            for i, p_err in enumerate(p_error_list):
                tensorboard.add_scalar(
                    f"network/mean_q, p error {p_err}", mean_q_list[i], t
                )
                tensorboard.add_scalar(
                    f"network/success_rate, p error {p_err}", success_rate[i], t
                )
                tensorboard.add_scalar(
                    f"network/ground_state_rate, p error {p_err}",
                    ground_state_rate[i],
                    t,
                )

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("I'm alive my friend. I can see the shadows everywhere!")

    logger.info("Reach maximum number of training steps. Terminate!")
    msg = ("terminate", None)
    learner_io_queue.put(msg)

    save_model(policy_net, optimizer, criterion, save_model_path_date)
    logger.info(f"Saved policy network to {save_model_path_date}")

    tensorboard.close()
