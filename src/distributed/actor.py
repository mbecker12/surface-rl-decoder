from time import time, sleep
from collections import namedtuple
import logging
import numpy as np
from distributed.environment_set import EnvironmentSet
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION

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

    logger.info("Fire up all the environments!")

    env = SurfaceCode()  # TODO: need published gym environment here
    state_size = env.syndrome_size
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
    # TODO: why is dtype = (float, 3) in the lindeby code?
    local_buffer_qvalues = np.empty(
        (num_environments, size_local_memory_buffer), dtype=float
    )
    local_buffer_rewards = np.empty(
        (num_environments, size_local_memory_buffer), dtype=float
    )
    buffer_idx = 0

    actor_io_queue = args["actor_io_queue"]

    model = None

    performance_start = time()
    performance_stop = None

    priorities = np.empty((25, 128))  # priorities TODO probably for replay memory

    logger.info(f"Actor {actor_id} starting loop on device {device}")
    while True:
        sleep(0.3)
        steps_per_episode += 1

        # select action batch
        actions = np.random.randint(0, 4, size=(num_environments, 3))

        # generate a random terminal action somewhere
        if np.random.random_sample() < 0.3:
            terminate_index = np.random.randint(0, num_environments)
            actions[terminate_index][:] = (0, 0, TERMINAL_ACTION)

        q_values = np.random.random_sample(num_environments)
        next_states, rewards, terminals, _ = environments.step(actions)

        # next_states += np.random.randint(0, 255, size=next_states.shape, dtype=np.uint8)

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
            # TODO: get new weights for NN model here

            # this approach counts through all environments and local memory buffer continuously
            # with no differentiation between those two channels
            to_send = [
                *zip(local_buffer_transitions[:, :-1].flatten(), priorities.flatten())
            ]

            # to_send = (local_buffer_transitions[:, :-1], priorities.flatten())

            sleep(0.5)

            logger.info("Put data in actor_io_queue")
            actor_io_queue.put(to_send)
            buffer_idx = 0

        too_many_steps = steps_per_episode > size_action_history
        if np.any(terminals) or np.any(too_many_steps):
            # find terminal envs
            indices = np.argwhere(np.logical_or(terminals, too_many_steps)).flatten()

            reset_states = environments.reset_terminal_environments(indices=indices)
            next_states[indices] = reset_states[indices]
            steps_per_episode[indices] = 0

        states = next_states

        # performance_stop = time()
        # performance_elapsed = performance_stop - performance_start
        # print(f"{performance_elapsed=}")
        # performance_start = time()
