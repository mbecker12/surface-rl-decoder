"""
Main module to start the distributed multiprocessing setup
for reinforcement learning.
"""
import os
import traceback
from copy import deepcopy
import logging
import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from distributed.actor import actor
from distributed.learner import learner
from distributed.io import io_replay_memory
from distributed.model_util import save_metadata
from distributed.mp_util import configure_processes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-locals, too-many-statements
def start_vnet():
    logger.info("Configure Value Network Training...")
    (
        actor_args,
        mem_args,
        learner_args,
        env_args,
        global_config,
        queues,
    ) = configure_processes(rl_type="v")

    summary_path = env_args["summary_path"]
    summary_date = env_args["summary_date"]
    summary_run_info = env_args["summary_run_info"]
    code_size = env_args["code_size"]

    num_cuda_actors = env_args["num_cuda_actors"]
    num_actors = env_args["num_actors"]

    actor_io_queues = queues["actor_io_queues"]
    learner_actor_queues = queues["learner_actor_queues"]

    save_model_path = env_args["save_model_path"]
    model_name = env_args["model_name"]
    model_config = env_args["model_config"]
    model_config["rl_type"] = "v"
    learner_device = env_args["learner_device"]

    # set up tensorboard for monitoring
    tensorboard = SummaryWriter(
        os.path.join(summary_path, str(code_size), summary_date, summary_run_info)
    )
    tensorboard_string = "global config: " + str(global_config) + "\n"
    tensorboard.add_text("run_info/hyper_parameters", tensorboard_string)
    tensorboard.close()

    # # start processes
    logger.info("Prepare Processes...")
    # prepare the replay memory process
    io_process = mp.Process(target=io_replay_memory, args=(mem_args,))

    # prepare and start multiple actor processes
    actor_process = []
    for i in range(num_actors):
        if i < num_cuda_actors:
            actor_args["device"] = "cuda"
        else:
            actor_args["device"] = "cpu"

        actor_args["actor_io_queue"] = actor_io_queues[i]
        actor_args["learner_actor_queue"] = learner_actor_queues[i]
        actor_args["id"] = i
        actor_args["rl_type"] = "v"
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
            os.path.join(summary_path, str(code_size), summary_date, summary_run_info)
        )
        tensorboard.add_text("run_info/error_message", error_traceback)

        tensorboard.close()

    save_model_path_date_meta = os.path.join(
        save_model_path,
        str(code_size),
        summary_date,
        f"{model_name}_{code_size}_meta.yaml",
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
    start_vnet()
