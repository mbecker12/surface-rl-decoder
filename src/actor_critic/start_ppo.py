"""
Start the PPO learning algorithm
by initializing the PPO class."
"""
import logging
import os
from copy import deepcopy
import sys
import traceback
from distributed.mp_util import configure_processes
from actor_critic.ppo import PPO
from distributed.model_util import save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def start_ppo():
    """
    With the help of a utility function, all configuration
    is read from a config.ini file and the configuration parameters
    are provided in logically grouped dictionaries.
    With that, the PPO class is instantiated
    and its train() function is called to start the PPO
    learning process.
    """
    logger.info("Configure PPO process")
    (
        worker_args,
        mem_args,
        learner_args,
        env_args,
        global_config,
        queues,
    ) = configure_processes(rl_type="ppo")

    logger.info("Set up PPO environment")
    ppo_agent = PPO(
        worker_args, mem_args, learner_args, env_args, global_config, queues
    )

    seed = worker_args.get("seed", 0)
    logger.info("Prepare training")
    try:
        ppo_agent.train(seed)
    except Exception as err:
        msg = ("terminate", None)
        for queue in ppo_agent.worker_queues:
            queue[0].send(msg)

        error_traceback = traceback.format_exc()
        logger.error("Caught exception in learning step")
        logger.error(error_traceback)

        logger.info("Saving Metadata")
        metadata = {}
        metadata["global"] = deepcopy(global_config)
        model_config = env_args["model_config"]
        model_name = env_args["model_name"]
        metadata["network"] = deepcopy(model_config)
        metadata["network"]["name"] = model_name
        save_model_path = env_args["save_model_path"]
        code_size = env_args["code_size"]
        summary_date = env_args["summary_date"]

        save_model_path_date_meta = os.path.join(
            save_model_path,
            str(code_size),
            summary_date,
            f"{model_name}_{code_size}_meta.yaml",
        )
        save_metadata(metadata, save_model_path_date_meta)
        sys.exit()


if __name__ == "__main__":
    start_ppo()
