from time import time, sleep

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learner")
logger.setLevel(logging.INFO)


def learner(args):
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    verbosity = args["verbosity"]

    start_time = time()
    max_time = 1000

    heart = time()
    heartbeat_interval = 10  # seconds
    timesteps = 1000

    for _ in range(timesteps):
        if time() - start_time > max_time:
            logger.info("Learner: time exceeded, aborting...")
            break

        if io_learner_queue.qsize == 0:
            logger.info("Learner waiting")

        data = io_learner_queue.get()
        if data is not None:
            logger.info(f"{len(data)=}")

        p_update = ([7624], [9866])  # just put something here for now
        msg = ("priorities", p_update)
        learner_io_queue.put(msg)

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.info("I'm alive my friend. I can see the shadows everywhere!")

        sleep(2)

    msg = ("terminate", None)
    learner_io_queue.put(msg)
