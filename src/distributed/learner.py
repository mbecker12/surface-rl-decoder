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
from distributed.util import time_ms

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
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
    max_time = max_time_h * 60 * 60  # seconds

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
    tensorboard = SummaryWriter(
        os.path.join(summary_path, str(code_size), summary_date, "learner")
    )
    tensorboard_step = 0
    received_data = 0

    # start the actual learning
    t = 0  # no worries, t gets incremented at the end of the while loop
    perfromance_start = time()
    eval_step = 0
    while t < timesteps:
        current_time = time()
        current_time_ms = time_ms()
        delta_t = current_time - perfromance_start
        count_to_eval += 1

        if time() - start_time > max_time:
            logger.warning("Learner: time exceeded, aborting...")
            break

        # after a certain number of steps, update the frozen target network
        if t % target_update_steps == 0 and t > 0:
            logger.debug("Update target network parameters")
            update_target_net_start = time()
            params = parameters_to_vector(policy_net.parameters())
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device)
            if benchmarking:
                update_target_net_stop = time()
                logger.debug(
                    "Time for updating target net parameters: "
                    f"{update_target_net_stop - update_target_net_start} s."
                )

            # notify the actor process that its network parameters should be updated
            msg = ("network_update", params.detach())
            logger.debug("Send network weights to actor process")
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

            if verbosity >= 3:
                tensorboard.add_scalar(
                    "learner/received_data",
                    received_data,
                    delta_t,
                    walltime=current_time_ms,
                )
                tensorboard_step += 1

        # perform the actual learning
        try:
            learning_step_start = time()
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
            )

            if benchmarking and t % eval_frequency == 0:
                learning_step_stop = time()
                logger.info(
                    f"Time for q-learning step: {learning_step_stop - learning_step_start} s."
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
            logger.info(f"Start Evaluation, Step {t+1}")
            count_to_eval = 0

            evaluation_start = time()
            episode_results, step_results, p_error_results = evaluate(
                policy_net,
                "",
                device,
                p_error_list,
                p_msmt_list,
                plot_one_episode=False,
                epsilon=learner_epsilon,
            )
            if benchmarking:
                evaluation_stop = time()
                logger.info(
                    f"Time for evaluation: {evaluation_stop - evaluation_start} s."
                )

            episode_results = transform_list_dict(episode_results)
            step_results = transform_list_dict(step_results)
            p_error_results = transform_list_dict(p_error_results)

            if verbosity:
                log_evaluation_data(
                    tensorboard,
                    p_error_list,
                    episode_results,
                    step_results,
                    p_error_results,
                    eval_step,
                    current_time_ms,
                )

            eval_step += 1

            # monitor policy network parameters
            if verbosity >= 4:
                policy_params = list(policy_net.parameters())
                n_layers = len(policy_params)
                for i, param in enumerate(policy_params):
                    if i == 0:
                        first_layer_params = param.detach().cpu().numpy()
                        tensorboard.add_histogram(
                            "learner/first_layer",
                            first_layer_params.reshape(-1, 1),
                            tensorboard_step,
                            walltime=current_time_ms,
                        )

                    if i == n_layers - 2:
                        last_layer_params = param.detach().cpu().numpy()
                        tensorboard.add_histogram(
                            "learner/last_layer",
                            last_layer_params.reshape(-1, 1),
                            tensorboard_step,
                            walltime=current_time_ms,
                        )

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("I'm alive my friend. I can see the shadows everywhere!")

        t += 1

    logger.info("Reach maximum number of training steps. Terminate!")
    msg = ("terminate", None)
    learner_io_queue.put(msg)

    save_model(policy_net, optimizer, criterion, save_model_path_date)
    logger.info(f"Saved policy network to {save_model_path_date}")

    tensorboard.close()
