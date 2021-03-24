"""
Define the actor process for exploration of the environment in
reinforcement learning.
"""
import os
from time import time
from collections import namedtuple
import logging
import numpy as np

# pylint: disable=not-callable
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import vector_to_parameters
from environment_set import EnvironmentSet
from distributed.model_util import choose_model, extend_model_config, load_model
from distributed.util import anneal_factor, compute_priorities, select_actions, time_ms
from surface_rl_decoder.surface_code import SurfaceCode

# pylint: disable=too-many-statements,too-many-locals,too-many-branches

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "terminal"]
)


def actor(args):
    """
    Define the actor function to be run by a mp process.
    The actor defines multiple environments which are in differing states
    and can perform steps independent of each other.

    After a certain number of steps, the used policy network is updated
    with new parameters from the learner process.

    Parameters
    ==========
    args: dictionary containing actor configuration
        "actor_io_queue": mp.Queue object for communication between actor
            and replay memory
        "learner_actor_queue": mp.Queue object for communication between actor
            and learner process
        "num_environments": (int) number of independent environments to perform steps in
        "size_action_history": (int) maximum size of the action history of the environment,
            trying to execute more actions than this in one environment causes the environment
            to terminate and start again with a new syndrome.
        "size_local_memory_buffer":  (int) maximum number of objects in the local
            memory store for transitions, actions, q values, rewards
        "num_actions_per_qubit": (int) number of possible operators on a qubit,
            default should be 3, for Pauli-X, -Y, -Z
        "verbosity": verbosity level
        "epsilon": (float) probability to choose a random action
        "model_name": (str) specifier for the model
        "model_config": (dict) configuration for network architecture.
            May change with different architectures
        "benchmarking": whether certain performance time measurements should be performed
        "summary_path": (str), base path for tensorboard
        "summary_date": (str), target path for tensorboard for current run
    """
    num_environments = args["num_environments"]
    actor_id = args["id"]
    size_action_history = args["size_action_history"]
    device = args["device"]
    verbosity = args["verbosity"]
    benchmarking = args["benchmarking"]
    num_actions_per_qubit = args["num_actions_per_qubit"]
    epsilon = args["epsilon"]
    load_model_flag = args["load_model"]
    old_model_path = args["old_model_path"]
    discount_factor = args["discount_factor"]
    discount_intermediate_reward = float(args.get("discount_intermediate_reward", 0.75))
    min_value_factor_intermediate_reward = float(
        args.get("min_value_intermediate_reward", 0.0)
    )
    decay_factor_intermediate_reward = float(
        args.get("decay_factor_intermediate_reward", 1.0)
    )
    decay_factor_epsilon = float(args.get("decay_factor_epsilon", 1.0))
    min_value_factor_epsilon = float(args.get("min_value_factor_epsilon", 0.0))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("actor")
    if verbosity >= 4:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Fire up all the environments!")

    env = SurfaceCode()
    state_size = env.syndrome_size
    code_size = state_size - 1
    stack_depth = env.stack_depth

    # create a collection of independent environments
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

    # initialize all states
    states = environments.reset_all()
    steps_per_episode = np.zeros(num_environments)

    # initialize local memory buffers
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

    # load communication queues
    actor_io_queue = args["actor_io_queue"]
    learner_actor_queue = args["learner_actor_queue"]

    # initialize the policy agent
    model_name = args["model_name"]
    model_config = args["model_config"]
    model_config = extend_model_config(
        model_config, state_size, stack_depth, device=device
    )
    model = choose_model(model_name, model_config)

    if load_model_flag:
        model, _, _ = load_model(model, old_model_path)
        logger.info(f"Loaded actor model from {old_model_path}")

    model.to(device)

    performance_start = time()
    heart = time()
    heartbeat_interval = 60  # seconds

    logger.info(f"Actor {actor_id} starting loop on device {device}")
    sent_data_chunks = 0

    # initialize tensorboard for monitoring/logging
    summary_path = args["summary_path"]
    summary_date = args["summary_date"]
    tensorboard = SummaryWriter(
        os.path.join(summary_path, str(code_size), summary_date, "actor")
    )
    tensorboard_step = 0

    # start the main exploration loop
    while True:
        steps_per_episode += 1

        # select actions based on the chosen model and latest states
        _states = torch.tensor(states, dtype=torch.float32, device=device)
        select_action_start = time()
        current_time_ms = time_ms()
        delta_t = select_action_start - performance_start

        annealed_epsilon = anneal_factor(
            delta_t,
            decay_factor=decay_factor_epsilon,
            min_value=min_value_factor_epsilon,
            base_factor=epsilon,
        )

        actions, q_values = select_actions(
            _states, model, state_size - 1, epsilon=annealed_epsilon
        )

        if benchmarking:
            select_action_stop = time()
            logger.info(
                f"time for select action: {select_action_stop - select_action_start}"
            )

        if verbosity >= 2:
            tensorboard.add_scalars(
                "actor/epsilon",
                {"annealed_epsilon": annealed_epsilon},
                delta_t,
                walltime=current_time_ms,
            )

        # perform the chosen actions
        steps_start = time()

        annealing_intermediate_reward = anneal_factor(
            delta_t,
            decay_factor=decay_factor_intermediate_reward,
            min_value=min_value_factor_intermediate_reward,
        )
        next_states, rewards, terminals, _ = environments.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=0,
        )

        if benchmarking:
            steps_stop = time()
            logger.info(
                f"time to step through environments: {steps_stop - steps_start}"
            )

        if verbosity >= 2:
            current_time_ms = time_ms()
            tensorboard.add_scalars(
                "actor/effect_intermediate_reward",
                {"anneal_factor": annealing_intermediate_reward},
                delta_t,
                walltime=current_time_ms,
            )

        # save transitions to local buffer
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

        # prepare to send local transitions to replay memory
        if buffer_idx >= size_local_memory_buffer:
            # get new weights for the policy model here
            if (learner_qsize := learner_actor_queue.qsize()) > 0:
                # consume all the deprecated updates without effect
                for _ in range(learner_qsize - 1):
                    learner_actor_queue.get()
                msg, network_params = learner_actor_queue.get()
                assert msg is not None
                assert network_params is not None
                if msg == "network_update":
                    logger.debug(
                        "Received new network weights. "
                        f"Taken the latest of {learner_qsize} updates."
                    )
                    vector_to_parameters(network_params, model.parameters())
                    model.to(device)

            new_local_qvalues = np.roll(local_buffer_qvalues, -1, axis=1)
            priorities = compute_priorities(
                local_buffer_actions[:, :-1],
                local_buffer_rewards[:, :-1],
                local_buffer_qvalues[:, :-1],
                new_local_qvalues[:, :-1],
                discount_factor,
                code_size,
            )

            # this approach counts through all environments and local memory buffer continuously
            # with no differentiation between those two channels
            to_send = [
                *zip(local_buffer_transitions[:, :-1].flatten(), priorities.flatten())
            ]

            logger.debug("Put data in actor_io_queue")
            actor_io_queue.put(to_send)
            if verbosity >= 4:
                sent_data_chunks += buffer_idx
                current_time_ms = time_ms()
                tensorboard.add_scalar(
                    "actor/sent_data_chunks",
                    sent_data_chunks,
                    delta_t,
                    walltime=current_time_ms,
                )
                tensorboard_step += 1

            buffer_idx = 0

        # determine episodes which are to be deemed terminal
        too_many_steps = steps_per_episode > size_action_history
        if np.any(terminals) or np.any(too_many_steps):
            # find terminal envs
            indices = np.argwhere(np.logical_or(terminals, too_many_steps)).flatten()

            reset_states = environments.reset_terminal_environments(indices=indices)
            next_states[indices] = reset_states[indices]
            steps_per_episode[indices] = 0

        # update states for next iteration
        states = next_states

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.debug("It's alive, can you feel it?")
