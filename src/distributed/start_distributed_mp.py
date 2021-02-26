from time import sleep
import logging
import multiprocessing as mp
from distributed.actor import actor
from distributed.learner import learner
from distributed.io import replay_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def start_mp():
    logger.info("Initialize queues")
    actor_io_queue = mp.Queue()
    learner_io_queue = mp.Queue()
    io_learner_queue = mp.Queue()
    sleep(2)
    mem_args = {
        "foo": "bar",
        "spam": "ham",
        "actor_io_queue": actor_io_queue,
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
    }

    io_process = mp.Process(target=replay_memory, args=(mem_args,))

    actor_args = {"marco": "polo", "uno": "dos", "actor_io_queue": actor_io_queue}
    actor_process = mp.Process(target=actor, args=(actor_args,))
    logger.info("Spawn actor process")
    actor_process.start()

    logger.info("Spawn io process")
    io_process.start()

    learner_args = {
        "baz": "bar",
        "hallo": "echo",
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
    }
    logger.info("Start learner")
    learner(learner_args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start_mp()
