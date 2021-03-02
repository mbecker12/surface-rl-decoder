from time import time, sleep
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("io")
logger.setLevel(logging.INFO)

# TODO: find out if this is really the replay memory
def replay_memory(args):
    heart = time()
    heartbeat_interval = 10

    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    actor_io_queue = args["actor_io_queue"]
    batch_in_queue_limit = 5

    while True:
        sleep(3)
        # empty queue of transtions from actors
        while not actor_io_queue.empty():

            # explainer for indices of transitions
            # [n_environment][n local memory buffer][states, actions, rewards, next_states, terminals]
            transitions, _ = actor_io_queue.get()
            for i, _ in enumerate(transitions):
                assert transitions[i][0][0].shape == (8, 6, 6), transitions[i][0][0].shape
                assert transitions[i][0][1].shape == (3, ), transitions[i][0][1].shape
                assert isinstance(transitions[i][0][2], (float, np.float64, np.float32)), type(transitions[i][0][2])
                assert transitions[i][0][3].shape == (8, 6, 6), transitions[i][0][3].shape
                assert isinstance(transitions[i][0][4], (bool, np.bool_)), type(transitions[i][0][4])
                assert transitions[i][1], transitions[i][1]
                if i == 0:
                    # logger.info(f"{transitions[i]=}")
                    logger.info(f"{transitions[i].shape=}")
                    assert transitions[i][0].shape == transitions[i][1].shape

            logger.info("Transitions look fine")
            

            

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
