import os
from time import time, sleep
from collections import namedtuple
import logging
import numpy as np
import torch
from distributed.environment_set import EnvironmentSet
from learner import learner
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION
from dummy_agent import DummyModel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import vector_to_parameters
from util import select_actions

# pylint: disable=too-many-statements,too-many-locals

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "terminal"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor")
logger.setLevel(logging.INFO)


def actor(args):
    num_environments = args["num_environments"]
    actor_id = args["id"]
    size_action_history = args["size_action_history"]
    device = args["device"]
    verbosity = args["verbosity"]
    benchmarking = args["benchmarking"]
    num_actions_per_qubit = args["num_actions_per_qubit"]

    logger.info("Fire up all the environments!")

    env = SurfaceCode()  # TODO: need published gym environment here
    state_size = env.syndrome_size
    code_size = state_size - 1
    stack_depth = env.stack_depth

    environments = EnvironmentSet(env, num_environments)

    transition_type = np.dtype(
        [
            ("state", (np.uint8, (stack_depth, state_size, state_size))),
            ("action", (np.uint8, 3)),
            ("reward", float),
            ("next_state", (np.uint8, (stack_depth, state_size, state_size))),
            ("terminal", bool),
        ]
    )

    states = environments.reset_all()
    steps_per_episode = np.zeros(num_environments)

    size_local_memory_buffer = args["size_local_memory_buffer"] + 1
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

    actor_io_queue = args["actor_io_queue"]
    learner_actor_queue = args["learner_actor_queue"]

    model = DummyModel(state_size, stack_depth)

    performance_start = time()
    performance_stop = None

    priorities = np.empty((25, 128))  # priorities TODO probably for replay memory

    logger.info(f"Actor {actor_id} starting loop on device {device}")
    sent_data_chunks = 0

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]
    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "actor"))
    tensorboard_step = 0
    while True:
        sleep(1)
        steps_per_episode += 1

        _states = torch.tensor(states, dtype=torch.float32)

        start_select_action = time()
        actions, q_values = select_actions(_states, model, state_size - 1)
        if benchmarking:
            logger.info(f"time for select action: {time() - start_select_action}")

        start_steps = time()
        next_states, rewards, terminals, _ = environments.step(actions)
        if benchmarking:
            logger.info(f"time to step through environments: {time() - start_steps}")

        transitions = np.asarray(
            [
                Transition(
                    states[i], actions[i], rewards[i], next_states[i], terminals[i]
                )
                for i in range(num_environments)
            ],
            dtype=transition_type,
        )

        local_buffer_transitions[:, buffer_idx] = transitions
        local_buffer_actions[:, buffer_idx] = actions
        local_buffer_qvalues[:, buffer_idx] = q_values
        local_buffer_rewards[:, buffer_idx] = rewards
        buffer_idx += 1

        if buffer_idx >= size_local_memory_buffer:
            # get new weights for the policy model here
            if learner_actor_queue.qsize() > 0:
                msg, network_params = learner_actor_queue.get()
                if msg == "network_update":
                    logger.info("Received new network weights")
                    vector_to_parameters(network_params, model.parameters())
                    model.to(device)

            # this approach counts through all environments and local memory buffer continuously
            # with no differentiation between those two channels
            to_send = [
                *zip(local_buffer_transitions[:, :-1].flatten(), priorities.flatten())
            ]

            sleep(0.2)

            logger.info("Put data in actor_io_queue")
            actor_io_queue.put(to_send)
            if verbosity:
                sent_data_chunks += buffer_idx
                tensorboard.add_scalar(
                    "actor/actions", sent_data_chunks, tensorboard_step
                )
                tensorboard_step += 1

            buffer_idx = 0

        too_many_steps = steps_per_episode > size_action_history
        if np.any(terminals) or np.any(too_many_steps):
            # find terminal envs
            indices = np.argwhere(np.logical_or(terminals, too_many_steps)).flatten()

            reset_states = environments.reset_terminal_environments(indices=indices)
            next_states[indices] = reset_states[indices]
            steps_per_episode[indices] = 0

        states = next_states
