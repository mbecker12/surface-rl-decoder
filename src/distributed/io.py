import os
from time import time, sleep
import logging
import numpy as np
from distributed.replay_memory import ReplayMemory
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("io")
logger.setLevel(logging.INFO)


# TODO: find out if this is really the replay memory
def io_replay_memory(args):
    heart = time()
    heartbeat_interval = 10

    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    actor_io_queue = args["actor_io_queue"]
    batch_in_queue_limit = 10
    verbosity = args["verbosity"]

    n_transitions_total = 0
    memory_size = args["replay_memory_size"]
    replay_size_before_sampling = args["replay_size_before_sampling"]

    replay_memory = ReplayMemory(memory_size)
    batch_size = args["batch_size"]

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]

    start_learning = False

    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "io"))
    tensorboard_step = 0

    count_consumption_outgoing = (
        0  # count the consumption of batches to be sent to the learner process
    )
    count_transition_received = 0  # count the number of transitions that arrived in this io module from the actor process
    consumption_total = 0
    transitions_total = 0
    stop_watch = time()
    while True:
        # sleep(1)
        # process the transitions sent from the actor process
        while not actor_io_queue.empty():

            # explainer for indices of transitions
            # [n_environment][n local memory buffer][states, actions, rewards, next_states, terminals]
            transitions = actor_io_queue.get()
            for i, _ in enumerate(transitions):
                assert transitions[i] is not None
                _transitions, _priorities = transitions[i]

                ## if the zip method is chosen in the actor
                assert _transitions[0].shape == (8, 6, 6), _transitions[0].shape
                assert _transitions[1].shape == (3,), _transitions[1].shape
                assert isinstance(
                    _transitions[2], (float, np.float64, np.float32)
                ), type(_transitions[2])
                assert _transitions[3].shape == (8, 6, 6), _transitions[3].shape
                assert isinstance(_transitions[4], (bool, np.bool_)), type(
                    _transitions[4]
                )

                replay_memory.save((_transitions, _priorities))
                n_transitions_total += 1
                count_transition_received += 1

                if i == 0 and verbosity:
                    tensorboard.add_scalar(
                        "transition/reward", _transitions[2], tensorboard_step
                    )
                    tensorboard.add_scalars(
                        "transition/action",
                        {
                            "x": _transitions[1][0],
                            "y": _transitions[1][1],
                            "action": _transitions[1][2],
                        },
                        tensorboard_step,
                    )

                    if verbosity > 2:
                        transition_shape = _transitions[0][-1].shape

                        _state_float = _transitions[0][-1].astype(np.float32)
                        _next_state_float = _transitions[3][-1].astype(np.float32)

                        tensorboard.add_image(
                            "transition/state",
                            _state_float,
                            tensorboard_step,
                            dataformats="HW",
                        )
                        tensorboard.add_image(
                            "transition/next_state",
                            _next_state_float,
                            tensorboard_step,
                            dataformats="HW",
                        )
                        action_matrix = np.zeros(
                            (transition_shape[0] - 1, transition_shape[1] - 1),
                            dtype=np.float32,
                        )
                        action = _transitions[1]
                        action_matrix[action[0], action[1]] = action[-1] / max(
                            TERMINAL_ACTION, 3
                        )
                        tensorboard.add_image(
                            "transition/action_viz",
                            action_matrix,
                            tensorboard_step,
                            dataformats="HW",
                        )

            logger.info("Saved transitions to replay memory")

        # TODO: log gpu metrics here
        if verbosity:
            transitions_total += count_transition_received
            consumption_total += count_consumption_outgoing

            current_time = time()
            tensorboard.add_scalars(
                "data/total",
                {
                    "total batch consumption outgoing": consumption_total,
                    "total # received transitions": transitions_total,
                },
                tensorboard_step,
            )
            tensorboard.add_scalars(
                "data/speed",
                {
                    "batch consumption rate of outgoing transitions": count_consumption_outgoing
                    / (current_time - stop_watch),
                    "received transitions rate": count_transition_received
                    / (current_time - stop_watch),
                },
                tensorboard_step,
            )

            count_transition_received = 0
            count_consumption_outgoing = 0
            tensorboard_step += 1
            stop_watch = time()

        if (
            not start_learning
        ) and replay_memory.current_num_objects >= replay_size_before_sampling:
            start_learning = True
            logger.info("Start Learning")
        else:
            sleep(1)

        while start_learning and (io_learner_queue.qsize() < batch_in_queue_limit):
            # TODO: lindeby returns
            # transitions, weights, indices, priorities
            # what are the different elements?
            transitions, memory_weights, indices, priorities = replay_memory.sample(
                batch_size
            )
            data = (transitions, memory_weights, indices)
            logger.info(f"{io_learner_queue.qsize()=}")
            logger.info(f"{replay_memory.current_num_objects=}")
            io_learner_queue.put(data)
            logger.info("Put data in io_learner_queue")

            count_consumption_outgoing += batch_size

        # check if the queue from the learner is empty
        terminate = False
        while not learner_io_queue.empty():

            msg, item = learner_io_queue.get()

            if msg == "priorities":
                # Update priorities
                logger.info("received message 'priorities' from learner")
                # logger.info(f"{item=}")
            elif msg == "terminate":
                logger.info("received message 'terminate' from learner")
                logger.info(
                    f"Total amount of generated transitions: {n_transitions_total}"
                )

        if time() - heart > heartbeat_interval:
            heart = time()
