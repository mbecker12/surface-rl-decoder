"""
Start the PPO learning algorithm
by initializing the PPO class."
"""
import logging
import sys
import traceback
from distributed.mp_util import configure_processes
from actor_critic.ppo import PPO

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
        sys.exit()

if __name__ == "__main__":
    start_ppo()
