from time import time, sleep
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("io")
logger.setLevel(logging.INFO)


def replay_memory(args):
    heart = time()
    heartbeat_interval = 10

    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    actor_io_queue = args["actor_io_queue"]
    batch_in_queue_limit = 5

    while True:
        sleep(5)
        # empty queue of transtions from actors
        while not actor_io_queue.empty():

            transitions = actor_io_queue.get()
            logger.info(f"{transitions[0]=}")

        while io_learner_queue.qsize() < batch_in_queue_limit:
            data = np.empty((8, 10))
            io_learner_queue.put(data)
            logger.info("put data in io_learner_queue")

        # empty queue from learner
        terminate = False
        while not learner_io_queue.empty():

            msg, item = learner_io_queue.get()

            if msg == "priorities":
                # Update priorities
                logger.info("received message 'priorities'")
                logger.info(f"{item=}")
            elif msg == "terminate":
                logger.info("received message 'terminate'")

        if time() - heart > heartbeat_interval:
            heart = time()
