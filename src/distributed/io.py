"""
Implementation of the IO process to handle replay memory.
Connects to the actor and learner process to store and share data
between those processes.
"""

import os
from time import time, sleep
import logging
import numpy as np
from distributed.replay_memory import ReplayMemory
from io_util import (
    add_transition_images_to_tensorboard,
    assert_transition_shapes,
    handle_transition_monitoring,
    monitor_cpu_memory,
    monitor_data_io,
    monitor_gpu_memory,
)
from prioritized_replay_memory import PrioritizedReplayMemory
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION
import torch
from torch.utils.tensorboard import SummaryWriter
import nvgpu
import psutil

from util import anneal_factor, time_ms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("io")
logger.setLevel(logging.INFO)


def io_replay_memory(args):
    """
    Start an instance of the replay memory process.
    Receives transitions from the actor processes and stores them
    in a replay-memory object.
    Upon request it will sample and send the replay memories to the learner process.

    Parameters
    ==========
    args: (dict)
        "actor_io_queue": mp.Queue object to communicate between actor and io module
        "learner_io_queue": mp.Queue object to communicate between learner and io module
        "io_learner_queue": mp.Queue object to communicate between io module and learner
        "replay_memory_size": (int) storage size (num of objects) of this replay memory instance
        "replay_size_before_sampling": (int) number of elements to accumulate before a
            meaningful sample will be generated
        "batch_size": (int) number of elements in one batch that gets sent to the learner process
        "stack_depth": (int), number of layers in syndrome stack
        "syndrome_size": (int), dimension of syndrome/state layer, usually code_distance + 1
        "verbosity": (int) verbosity level
        "benchmarking": (int/bool) whether or not to perform certain timing actions for benchmarking
        "summary_path": (str), base path for tensorboard
        "summary_date": (str), target path for tensorboard for current run
    """
    heart = time()
    heartbeat_interval = 60  # seconds

    # initialization
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    actor_io_queue = args["actor_io_queue"]
    batch_in_queue_limit = 10
    verbosity = args["verbosity"]

    n_transitions_total = 0
    memory_size = args["replay_memory_size"]
    memory_alpha = float(args["replay_memory_alpha"])
    memory_beta = float(args["replay_memory_beta"])
    decay_factor_beta = float(args.get("replay_memory_decay_beta"))
    replay_size_before_sampling = args["replay_size_before_sampling"]

    memory_type = args["replay_memory_type"]

    if memory_type.lower() == "uniform":
        replay_memory = ReplayMemory(memory_size)
    elif "prio" in memory_type.lower() or "per" in memory_type.lower():
        replay_memory = PrioritizedReplayMemory(memory_size, memory_alpha)
    else:
        raise Exception(f"Error! Memory type '{memory_type}' not supported.")

    logger.info(
        f"Initialized replay memory of type {memory_type}, an instance of {type(replay_memory).__name__}."
    )

    batch_size = args["batch_size"]
    stack_depth = args["stack_depth"]
    syndrome_size = args["syndrome_size"]

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]

    nvidia_log_frequency = args["nvidia_log_frequency"]

    start_learning = False

    # initialize tensorboard for monitoring/logging
    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "io"))
    tensorboard_step = 0

    # prepare data throughput metrics
    count_consumption_outgoing = (
        0  # count the consumption of batches to be sent to the learner process
    )
    count_transition_received = 0  # count the number of transitions that arrived in this io module from the actor process
    consumption_total = 0
    transitions_total = 0

    stop_watch = time()
    performance_start = time()
    nvidia_log_time = time()
    try:
        gpu_info = nvgpu.gpu_info()
        gpu_available = True
    except FileNotFoundError as _:
        gpu_available = False

    while True:

        # process the transitions sent from the actor process
        while not actor_io_queue.empty():

            # explainer for indices of transitions
            # [n_environment][n local memory buffer][states, actions, rewards, next_states, terminals]
            transitions = actor_io_queue.get()
            if verbosity and (transitions is not None):
                random_sample_indices = np.random.choice(range(len(transitions)), 10)
                if verbosity >= 2:
                    priority_sample = np.zeros(len(transitions))
            for i, _ in enumerate(transitions):
                assert transitions[i] is not None
                _transitions, _priorities = transitions[i]
                assert_transition_shapes(_transitions, stack_depth, syndrome_size)

                # save transitions to the replay memory store
                replay_memory.save(_transitions, _priorities)

                n_transitions_total += 1
                count_transition_received += 1

                if verbosity and (i in random_sample_indices):
                    current_time_ms = time_ms()
                    handle_transition_monitoring(
                        tensorboard,
                        _transitions,
                        verbosity,
                        tensorboard_step,
                        current_time_ms,
                        TERMINAL_ACTION,
                    )
                    tensorboard_step += 1

                if verbosity >= 2:
                    assert isinstance(
                        _priorities,
                        float,
                    ), f"{type(_priorities)=}"
                    assert _priorities > 0, f"{_priorities=}"
                    priority_sample[i] = _priorities
                # end if; logging
            # end for loop; transitions
            if verbosity >= 2:
                received_priorities = np.array(priority_sample, dtype=np.float32)
                percentile = np.percentile(received_priorities, 95)
                received_priorities_idx = np.where(received_priorities < percentile)
                received_priorities = received_priorities[received_priorities_idx]
                if len(received_priorities) > 0:
                    tensorboard.add_histogram(
                        "io/received_priorities",
                        received_priorities,
                        tensorboard_step,
                        walltime=current_time_ms,
                    )

            logger.debug("Saved transitions to replay memory")

            if verbosity:
                current_time = time()
                current_time_ms = time_ms()

                # log gpu stats
                if gpu_available and nvidia_log_time > nvidia_log_frequency:
                    monitor_gpu_memory(
                        tensorboard, current_time, performance_start, current_time_ms
                    )
                if verbosity >= 2:
                    monitor_cpu_memory(
                        tensorboard, current_time, performance_start, current_time_ms
                    )

                # log data churning
                transitions_total += count_transition_received
                consumption_total += count_consumption_outgoing
                monitor_data_io(
                    tensorboard,
                    consumption_total,
                    transitions_total,
                    count_consumption_outgoing,
                    count_transition_received,
                    stop_watch,
                    current_time,
                    performance_start,
                    current_time_ms,
                )

                count_transition_received = 0
                count_consumption_outgoing = 0
                stop_watch = time()
        # end while; actor_io_queue

        # if the replay memory is sufficiently filled, trigger the actual learning
        if (
            not start_learning
        ) and replay_memory.filled_size() >= replay_size_before_sampling:
            start_learning = True
            logger.info("Start Learning")
        else:
            sleep(1)

        # prepare to send data to learner process repeatedly
        while start_learning and (io_learner_queue.qsize() < batch_in_queue_limit):
            delta_t = time() - performance_start
            # want to anneal beta from ~0.4 to ~ 1,
            # so decay_factor should be larger than 1
            annealed_beta = anneal_factor(
                time_difference=delta_t,
                decay_factor=decay_factor_beta,
                max_value=1.0,
                base_factor=memory_beta,
            )

            transitions, memory_weights, indices, priorities = replay_memory.sample(
                batch_size, annealed_beta, tensorboard=tensorboard, verbosity=verbosity
            )

            current_time_ms = time_ms()
            if verbosity >= 2:
                tensorboard.add_scalars(
                    "io/beta (PER)",
                    {"beta": annealed_beta},
                    delta_t,
                    walltime=current_time_ms,
                )

                if priorities is not None:
                    sending_priorities = np.array(priorities, dtype=np.float32)
                    percentile = np.percentile(sending_priorities, 80)
                    sending_priorities_idx = np.where(sending_priorities < percentile)
                    sending_priorities = sending_priorities[sending_priorities_idx]
                    if len(sending_priorities) > 0:
                        tensorboard.add_histogram(
                            "io/sent_data_priorities",
                            sending_priorities,
                            delta_t,
                            walltime=current_time_ms,
                        )

            assert len(transitions) == batch_size
            data = (transitions, memory_weights, indices)
            logger.debug(f"{io_learner_queue.qsize()=}")
            logger.debug(f"{replay_memory.filled_size()=}")
            io_learner_queue.put(data)
            logger.debug("Put data in io_learner_queue")

            count_consumption_outgoing += batch_size

        # check if the queue from the learner is empty
        terminate = False
        while not learner_io_queue.empty():

            # look for priority updates for prioritized replay memory
            msg, item = learner_io_queue.get()

            if msg == "priorities":
                # Update priorities
                logger.debug("received message 'priorities' from learner")
                indices, priorities = item
                replay_memory.priority_update(indices, priorities)
            elif msg == "terminate":
                logger.info("received message 'terminate' from learner")
                tensorboard.close()
                logger.info(
                    f"Total amount of generated transitions: {n_transitions_total}"
                )

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("Oohoh I, ooh, I'm still alive")
