import multiprocessing as mp
import logging
from distributed.mp_util import configure_processes
from actor_critic.ppo import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def start_ppo():
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
    ppo_agent.train(seed)


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    start_ppo()
