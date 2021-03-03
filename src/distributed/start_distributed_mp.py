import os
import sys
from time import sleep
import logging
import multiprocessing as mp
from distributed.actor import actor
from distributed.learner import learner
from distributed.io import io_replay_memory
from iniparser import Config
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

SUMMARY_PATH = "runs"
# TODO: replace this with the actual date in the real setting
SUMMARY_DATE = "test"
SUMMARY_RUN_INFO = "run_info"


def start_mp():
    cfg = Config()
    cfg.scan(".", True).read()
    distributed_config = cfg.config_rendered.get("distributed_config")
    global_config = cfg.config_rendered.get("config")

    actor_config = distributed_config.get("actor")
    memory_config = distributed_config.get("replay_memory")
    learner_config = distributed_config.get("learner")

    env_config = global_config.get("env")  # TODO get env config as well
    general_config = global_config.get("general")

    batch_size = int(general_config["batch_size"])

    size_action_history = int(env_config.get("max_actions", "256"))

    num_cuda_actors = int(actor_config["num_cuda"])
    num_cpu_actors = int(actor_config["num_cpu"])
    num_actors = num_cuda_actors + num_cpu_actors
    num_environments = int(actor_config["num_environments"])
    size_local_memory_buffer = int(actor_config.get("size_local_memory_buffer"))
    actor_verbosity = int(actor_config["verbosity"])

    replay_memory_size = int(memory_config["size"])
    replay_size_before_sampling = int(memory_config["replay_size_before_sampling"])
    replay_memory_verbosity = int(memory_config["verbosity"])

    learner_verbosity = int(learner_config["verbosity"])

    # initialize queues
    logger.info("Initialize queues")
    actor_io_queue = mp.Queue()
    learner_io_queue = mp.Queue()
    io_learner_queue = mp.Queue()
    sleep(0.5)

    # configure processes
    mem_args = {
        "foo": "bar",
        "spam": "ham",
        "actor_io_queue": actor_io_queue,
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
        "replay_memory_size": replay_memory_size,
        "replay_size_before_sampling": replay_size_before_sampling,
        "batch_size": batch_size,
        "verbosity": replay_memory_verbosity,
        "summary_path": SUMMARY_PATH,
        "summary_date": SUMMARY_DATE,
    }

    actor_args = {
        "marco": "polo",
        "uno": "dos",
        "actor_io_queue": actor_io_queue,
        "num_environments": num_environments,
        "size_action_history": size_action_history,
        "size_local_memory_buffer": size_local_memory_buffer,
        "verbosity": actor_verbosity,
    }

    learner_args = {
        "baz": "bar",
        "hallo": "echo",
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
        "verbosity": learner_verbosity,
    }

    tensorboard = SummaryWriter(
        os.path.join(SUMMARY_PATH, SUMMARY_DATE, SUMMARY_RUN_INFO)
    )
    tensorboard_string = "global config: " + str(global_config) + "\n"
    tensorboard_string += "distributed config: " + str(distributed_config)
    tensorboard.add_text("run_info/hyper_parameters", tensorboard_string)
    tensorboard.close()

    # start processes
    io_process = mp.Process(target=io_replay_memory, args=(mem_args,))

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

    logger.info("Start learner")
    try:
        learner(learner_args)
    except Exception as _:
        # TODO: log the run here by using sys.exc_info()[0]
        tb = SummaryWriter(os.path.join(SUMMARY_PATH, SUMMARY_DATE, SUMMARY_RUN_INFO))
        tb.add_text("RunInfo/Error_Message", sys.exc_info()[0])
        tb.close()
        pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start_mp()
