from time import sleep
import logging
import multiprocessing as mp
from distributed.actor import actor
from distributed.learner import learner
from distributed.io import replay_memory
from iniparser import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def start_mp():
    cfg = Config()
    cfg.scan(".", True).read()
    distributed_config = cfg.config_rendered.get("distributed_config")
    environment_config = cfg.config_rendered.get("config")

    actor_config = distributed_config.get("actor")
    env_config = environment_config.get("env")  # TODO get env config as well

    size_action_history = int(env_config.get("max_actions", "256"))

    num_cuda_actors = int(actor_config["num_cuda"])
    num_cpu_actors = int(actor_config["num_cpu"])
    num_actors = num_cuda_actors + num_cpu_actors
    num_environments = int(actor_config["num_environments"])
    size_local_memory_buffer = int(actor_config.get("size_local_memory_buffer"))

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

    actor_args = {
        "marco": "polo",
        "uno": "dos",
        "actor_io_queue": actor_io_queue,
        "num_environments": num_environments,
        "size_action_history": size_action_history,
        "size_local_memory_buffer": size_local_memory_buffer,
    }
    actor_process = []
    for i in range(num_actors):
        if i < num_cuda_actors:
            actor_args["device"] = "cuda"
        else:
            actor_args["device"] = "cpu"

        actor_args["id"] = i
        actor_process.append(mp.Process(target=actor, args=(actor_args,)))
        logger.info(f"Spawn actor process {i}")
        actor_process[i].start()

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
