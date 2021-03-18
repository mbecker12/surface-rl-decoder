"""
Main module to start the distributed multiprocessing setup
for reinforcement learning.
"""
import os
import json
import traceback
from time import sleep
from copy import deepcopy
import logging
import multiprocessing as mp
from iniparser import Config
from torch.utils.tensorboard import SummaryWriter
from distributed.actor import actor
from distributed.learner import learner
from distributed.io import io_replay_memory
from model_util import save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

SUMMARY_PATH = "./runs"
# TODO: replace this with the actual date in the real setting
SUMMARY_DATE = "test2"
SUMMARY_RUN_INFO = "run_info"


def start_mp():
    """
    Start the actual sub processes.
    This will read in the configuration from available .ini files.
    Expect to find files containing the following sections:
        config
            env
            actor
            replay_memory
            learner

    The available configuration will determine the settings
    of the different subprocesses:
        replay memory
        actor (multiple)
        learner

    The communication between these processes is handled by
    Queue objects from the multiprocessing library.
    """

    # take care of all the configuration
    cfg = Config()
    cfg.scan(".", True).read()
    global_config = cfg.config_rendered.get("config")
    logger.info(f"\n{global_config=}\n\n")

    actor_config = global_config.get("actor")
    memory_config = global_config.get("replay_memory")
    learner_config = global_config.get("learner")

    # set up surface code environment configuration
    env_config = global_config.get("env")

    size_action_history = int(env_config.get("max_actions", "256"))
    system_size = int(env_config["size"])
    syndrome_size = system_size + 1
    stack_depth = int(env_config["stack_depth"])

    # set up actor configuration
    num_cuda_actors = int(actor_config["num_cuda"])
    num_cpu_actors = int(actor_config["num_cpu"])
    num_actors = num_cuda_actors + num_cpu_actors
    num_environments = int(actor_config["num_environments"])
    size_local_memory_buffer = int(actor_config.get("size_local_memory_buffer"))
    actor_verbosity = int(actor_config["verbosity"])
    actor_benchmarking = int(actor_config["benchmarking"])
    epsilon = float(actor_config["epsilon"])
    actor_load_model = int(actor_config["load_model"])
    num_actions_per_qubit = 3

    # set up replay memory configuration
    replay_memory_size = int(memory_config["size"])
    replay_size_before_sampling = int(memory_config["replay_size_before_sampling"])
    replay_memory_verbosity = int(memory_config["verbosity"])
    replay_memory_benchmarking = int(memory_config["benchmarking"])

    # set up learner configuration
    learner_verbosity = int(learner_config["verbosity"])
    learner_benchmarking = int(learner_config["benchmarking"])
    learner_max_time_h = int(learner_config["max_time_h"])
    learning_rate = float(learner_config["learning_rate"])
    learner_device = learner_config["device"]
    batch_size = int(learner_config["batch_size"])
    target_update_steps = int(learner_config["target_update_steps"])
    discount_factor = float(learner_config["discount_factor"])
    eval_frequency = int(learner_config["eval_frequency"])
    max_timesteps = int(learner_config["max_timesteps"])
    learner_epsilon = float(learner_config["learner_epsilon"])
    learner_eval_p_errors = [0.01, 0.02, 0.03]
    learner_eval_p_msmt = [0.01, 0.02, 0.03]
    learner_load_model = int(learner_config["load_model"])
    old_model_path = learner_config["load_model_path"]
    save_model_path = learner_config["save_model_path"]

    # initialize communication queues
    logger.info("Initialize queues")
    actor_io_queue = mp.Queue()
    learner_io_queue = mp.Queue()
    io_learner_queue = mp.Queue()
    learner_actor_queue = mp.Queue()

    model_name = learner_config["model_name"]
    model_config_location = learner_config["model_config_location"]
    model_config_file = learner_config["model_config_file"]
    model_config_file_path = os.path.join(model_config_location, model_config_file)

    # load json with potantially multiple model definitions
    with open(model_config_file_path) as json_file:
        model_config = json.load(json_file)

    # select the specification of the right model from the json
    model_config = model_config[model_name]

    # configure processes
    mem_args = {
        "actor_io_queue": actor_io_queue,
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
        "replay_memory_size": replay_memory_size,
        "replay_size_before_sampling": replay_size_before_sampling,
        "batch_size": batch_size,
        "stack_depth": stack_depth,
        "syndrome_size": syndrome_size,
        "verbosity": replay_memory_verbosity,
        "benchmarking": replay_memory_benchmarking,
        "summary_path": SUMMARY_PATH,
        "summary_date": SUMMARY_DATE,
    }

    actor_args = {
        "actor_io_queue": actor_io_queue,
        "learner_actor_queue": learner_actor_queue,
        "num_environments": num_environments,
        "size_action_history": size_action_history,
        "size_local_memory_buffer": size_local_memory_buffer,
        "num_actions_per_qubit": num_actions_per_qubit,
        "verbosity": actor_verbosity,
        "benchmarking": actor_benchmarking,
        "summary_path": SUMMARY_PATH,
        "summary_date": SUMMARY_DATE,
        "model_name": model_name,
        "model_config": model_config,
        "epsilon": epsilon,
        "load_model": actor_load_model,
        "old_model_path": old_model_path,
    }

    learner_args = {
        "syndrome_size": syndrome_size,
        "stack_depth": stack_depth,
        "learner_io_queue": learner_io_queue,
        "io_learner_queue": io_learner_queue,
        "learner_actor_queue": learner_actor_queue,
        "verbosity": learner_verbosity,
        "benchmarking": learner_benchmarking,
        "summary_path": SUMMARY_PATH,
        "summary_date": SUMMARY_DATE,
        "max_time": learner_max_time_h,
        "learning_rate": learning_rate,
        "device": learner_device,
        "target_update_steps": target_update_steps,
        "discount_factor": discount_factor,
        "batch_size": batch_size,
        "eval_frequency": eval_frequency,
        "learner_eval_p_error": learner_eval_p_errors,
        "learner_eval_p_msmt": learner_eval_p_msmt,
        "timesteps": max_timesteps,
        "model_name": model_name,
        "model_config": model_config,
        "learner_epsilon": learner_epsilon,
        "load_model": learner_load_model,
        "old_model_path": old_model_path,
        "save_model_path": save_model_path,
    }

    # set up tensorboard for monitoring
    tensorboard = SummaryWriter(
        os.path.join(SUMMARY_PATH, SUMMARY_DATE, SUMMARY_RUN_INFO)
    )
    tensorboard_string = "global config: " + str(global_config) + "\n"
    tensorboard.add_text("run_info/hyper_parameters", tensorboard_string)
    tensorboard.close()

    # # start processes

    # prepare the replay memory process
    io_process = mp.Process(target=io_replay_memory, args=(mem_args,))

    # prepare and start multiple actor processes
    actor_process = []
    for i in range(num_actors):
        if i < num_cuda_actors:
            actor_args["device"] = "cuda"
        else:
            actor_args["device"] = "cpu"

        actor_args["id"] = i
        actor_process.append(mp.Process(target=actor, args=(actor_args,)))
        logger.info(f"Spawn actor process {i} on device {actor_args['device']}")
        actor_process[i].start()

    # spawn replay memory process
    logger.info("Spawn io process")
    io_process.start()

    # spawn learner process
    logger.info(f"Start learner on device {learner_device}")
    try:
        learner(learner_args)
    # pylint: disable=broad-except
    except Exception as err:
        print(err)
        error_traceback = traceback.format_exc()
        logger.error("An error occurred!")
        logger.error(error_traceback)
        # log the actual error to the tensorboard
        tensorboard = SummaryWriter(
            os.path.join(SUMMARY_PATH, SUMMARY_DATE, SUMMARY_RUN_INFO)
        )
        tensorboard.add_text("run_info/error_message", error_traceback)

        tensorboard.close()

    save_model_path_date_meta = os.path.join(
        save_model_path,
        SUMMARY_DATE,
        f"{model_name}_{system_size}_{SUMMARY_DATE}_meta.yaml",
    )

    logger.info("Saving Metadata")
    metadata = {}
    metadata["global"] = deepcopy(global_config)
    metadata["network"] = deepcopy(model_config)
    metadata["network"]["name"] = model_name
    save_metadata(metadata, save_model_path_date_meta)

    logger.info("Training Done!")
    for i in range(num_actors):
        actor_process[i].terminate()
    io_process.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start_mp()
