from time import time, sleep
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor")
logger.setLevel(logging.INFO)


def actor(args):
    logger.info("This is where all the environments would fire up")
    actor_io_queue = args["actor_io_queue"]

    performance_start = time()
    performance_stop = None

    local_buffer = np.empty((10, 256))  # Transtions
    priorities = np.empty((25, 128))

    while True:
        sleep(0.5)

        performance_stop = time()
        performance_elapsed = performance_stop - performance_start

        performance_start = time()

        to_send = [*zip(local_buffer[:, :-1].flatten(), priorities.flatten())]
        actor_io_queue.put(to_send)

        logger.info(f"{performance_elapsed=}")
