import os
from time import time, sleep
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logging
from distributed.dummy_agent import DummyModel
from distributed.evaluate import evaluate
from distributed.learner_util import (
    parameters_to_vector,
    perform_q_learning_step,
    vector_to_parameters,
    data_to_batch,
    predict_max_optimized,
)

# from distributed.util import action_to_q_value_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learner")
logger.setLevel(logging.INFO)


def learner(args):
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    verbosity = args["verbosity"]

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
    count_to_eval = 0

    start_time = time()
    max_time_h = args["max_time"]  # hours
    max_time = max_time_h * 60 * 60  # seconds

    heart = time()
    heartbeat_interval = 10  # seconds
    timesteps = 10000

    policy_net = DummyModel(syndrome_size, stack_depth)
    target_net = DummyModel(syndrome_size, stack_depth)
    policy_net.to(device)
    target_net.to(device)
    optimizer = Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="none")

    received_data = 0

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]

    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "learner"))
    tensorboard_step = 0
    for t in range(timesteps):
        if time() - start_time > max_time:
            logger.warning("Learner: time exceeded, aborting...")
            break

        if t % target_update_steps == 0 and t > 0:
            logger.debug("Update target network parameters")
            params = parameters_to_vector(policy_net.parameters())
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device)

        if io_learner_queue.qsize == 0:
            logger.debug("Learner waiting")

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

        # update priorities in replay_memory
        p_update = (indices, priorities)
        msg = ("priorities", p_update)
        learner_io_queue.put(msg)

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
        count_to_eval += 1

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("I'm alive my friend. I can see the shadows everywhere!")
            if verbosity > 1:
                tensorboard.add_scalar("learner/heartbeat", 1, 0)

        sleep(1)

    logger.info("Time's up. Terminate!")
    msg = ("terminate", None)
    learner_io_queue.put(msg)

    # TODO: save model
    tensorboard.close()
