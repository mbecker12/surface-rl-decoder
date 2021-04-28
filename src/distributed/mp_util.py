"""
Utility function to set up configuration of subprocesses
"""
import os
import json
import logging
import multiprocessing as mp
import yaml
from iniparser import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")
logger.setLevel(logging.INFO)


def configure_processes(rl_type="q_learning"):
    # take care of all the configuration
    cfg = Config()
    cfg.scan(".", True).read()
    global_config = cfg.config_rendered.get("config")

    logger.info(
        "\nQEC Config: \n\n" f"{yaml.dump(global_config, default_flow_style=False)}"
    )

    actor_config = global_config.get("actor")
    memory_config = global_config.get("replay_memory")
    learner_config = global_config.get("learner")

    general_config = global_config.get("general")
    summary_path = general_config.get("summary_path", "runs")
    summary_date = general_config.get("summary_date", "test3")
    summary_run_info = general_config.get("summary_run_info", "run_info")

    # set up surface code environment configuration
    env_config = global_config.get("env")
    p_error = float(env_config.get("p_error", 0.01))
    p_msmt = float(env_config.get("p_msmt", 0.01))
    size_action_history = int(env_config.get("max_actions", "256"))
    code_size = int(env_config["size"])
    syndrome_size = code_size + 1
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
    discount_intermediate_reward = float(
        actor_config.get("discount_intermediate_reward", 0.3)
    )
    min_value_factor_intermediate_reward = float(
        actor_config.get("min_value_intermediate_reward", 0.0)
    )
    decay_factor_intermediate_reward = float(
        actor_config.get("decay_factor_intermediate_reward", 1.0)
    )
    decay_factor_epsilon = float(actor_config.get("decay_factor_epsilon", 1.0))
    min_value_factor_epsilon = float(actor_config.get("min_value_factor_epsilon", 0.0))
    seed = int(actor_config.get("seed", 0))

    # set up replay memory configuration
    replay_memory_size = int(memory_config["size"])
    replay_size_before_sampling = int(memory_config["replay_size_before_sampling"])
    replay_memory_verbosity = int(memory_config["verbosity"])
    replay_memory_benchmarking = int(memory_config["benchmarking"])
    replay_memory_type = memory_config["memory_type"]
    replay_memory_alpha = float(memory_config["alpha"])
    replay_memory_beta = float(memory_config["beta"])
    replay_memory_decay_beta = float(memory_config.get("decay_beta", 1.0))
    nvidia_log_frequency = int(memory_config.get("nvidia_log_frequency", 100))

    

    # set up learner configuration
    learner_verbosity = int(learner_config["verbosity"])
    learner_benchmarking = int(learner_config["benchmarking"])
    learner_max_time_h = float(learner_config["max_time_h"])
    learner_max_time_minutes = float(learner_config.get("max_time_minutes", 0.0))
    learning_rate = float(learner_config["learning_rate"])
    learner_device = learner_config["device"]
    batch_size = int(learner_config["batch_size"])
    target_update_steps = int(learner_config["target_update_steps"])
    discount_factor = float(learner_config["discount_factor"])
    eval_frequency = int(learner_config["eval_frequency"])
    max_timesteps = int(learner_config["max_timesteps"])
    learner_epsilon = float(learner_config["learner_epsilon"])
    learner_eval_p_errors = [p_error, p_error * 1.5]
    learner_eval_p_msmt = [p_msmt, p_msmt * 1.5]
    learner_load_model = int(learner_config["load_model"])
    old_model_path = learner_config["load_model_path"]
    save_model_path = learner_config["save_model_path"]

    # initialize communication queues
    logger.info("Initialize queues")
    if "q" in rl_type.lower():
        actor_io_queues = [None] * num_actors
        learner_actor_queues = [None] * num_actors
        for i in range(num_actors):
            actor_io_queues[i] = mp.Queue()
            learner_actor_queues[i] = mp.Queue()

        learner_io_queue = mp.Queue()
        io_learner_queue = mp.Queue()

        queues = {
            "actor_io_queues": actor_io_queues,
            "learner_io_queue": learner_io_queue,
            "io_learner_queue": io_learner_queue,
            "learner_actor_queues": learner_actor_queues,
        }
    elif "ppo" in rl_type.lower():
        worker_queues = [mp.Pipe() for i in range(num_actors)]

        queues = {"worker_queues": worker_queues}

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
        "replay_memory_size": replay_memory_size,
        "replay_size_before_sampling": replay_size_before_sampling,
        "batch_size": batch_size,
        "stack_depth": stack_depth,
        "syndrome_size": syndrome_size,
        "verbosity": replay_memory_verbosity,
        "benchmarking": replay_memory_benchmarking,
        "summary_path": summary_path,
        "summary_date": summary_date,
        "replay_memory_type": replay_memory_type,
        "replay_memory_alpha": replay_memory_alpha,
        "replay_memory_beta": replay_memory_beta,
        "replay_memory_decay_beta": replay_memory_decay_beta,
        "nvidia_log_frequency": nvidia_log_frequency,
    }

    actor_args = {
        "num_environments": num_environments,
        "size_action_history": size_action_history,
        "size_local_memory_buffer": size_local_memory_buffer,
        "num_actions_per_qubit": num_actions_per_qubit,
        "verbosity": actor_verbosity,
        "benchmarking": actor_benchmarking,
        "summary_path": summary_path,
        "summary_date": summary_date,
        "model_name": model_name,
        "model_config": model_config,
        "epsilon": epsilon,
        "load_model": actor_load_model,
        "old_model_path": old_model_path,
        "discount_factor": discount_factor,
        "discount_intermediate_reward": discount_intermediate_reward,
        "min_value_factor_intermediate_reward": min_value_factor_intermediate_reward,
        "decay_factor_intermediate_reward": decay_factor_intermediate_reward,
        "decay_factor_epsilon": decay_factor_epsilon,
        "min_value_factor_epsilon": min_value_factor_epsilon,
        "seed": seed,
    }

    learner_args = {
        "syndrome_size": syndrome_size,
        "stack_depth": stack_depth,
        "verbosity": learner_verbosity,
        "benchmarking": learner_benchmarking,
        "summary_path": summary_path,
        "summary_date": summary_date,
        "max_time": learner_max_time_h,
        "max_time_minutes": learner_max_time_minutes,
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

    env_args = {
        "p_error": p_error,
        "p_msmt": p_msmt,
        "size_action_history": size_action_history,
        "code_size": code_size,
        "syndrome_size": syndrome_size,
        "stack_depth": stack_depth,
        "summary_path": summary_path,
        "summary_date": summary_date,
        "summary_run_info": summary_run_info,
        "num_actors": num_actors,
        "num_cuda_actors": num_cuda_actors,
        "num_cpu_actors": num_cpu_actors,
        "save_model_path": save_model_path,
        "model_name": model_name,
        "model_config": model_config,
        "learner_device": learner_device,
    }

    if "ppo" in rl_type.lower():
        learner_args["policy_model_max_grad_norm"] = float(learner_config.get("policy_model_max_grad_norm"))
        learner_args["policy_clip_range"] = float(learner_config.get("policy_clip_range"))
        learner_args["policy_stopping_kl"] = float(learner_config.get("policy_stopping_kl"))
        learner_args["value_model_max_grad_norm"] = float(learner_config.get("value_model_max_grad_norm"))
        learner_args["value_clip_range"] = float(learner_config.get("value_clip_range"))
        learner_args["value_stopping_mse"] = float(learner_config.get("value_stopping_mse"))
        learner_args["entropy_loss_weight"] = float(learner_config.get("entropy_loss_weight"))
        learner_args["value_loss_weight"] = float(learner_config.get("value_loss_weight"))
        learner_args["max_episodes"] = int(learner_config.get("max_episodes"))
        learner_args["optimization_epochs"] = int(learner_config.get("optimization_epochs"))
        
        episode_buffer_tau = float(memory_config.get("episode_buffer_tau"))
        max_buffer_episodes = int(memory_config.get("max_buffer_episodes"))
        max_buffer_episode_steps = int(memory_config.get("max_buffer_episode_steps"))
        episode_buffer_device = memory_config.get("episode_buffer_device")

        mem_args["episode_buffer_tau"] = episode_buffer_tau
        mem_args["max_buffer_episodes"] = max_buffer_episodes
        mem_args["max_buffer_episode_steps"] = max_buffer_episode_steps
        mem_args["episode_buffer_device"] = episode_buffer_device

    if "q" in rl_type.lower():
        mem_args["actor_io_queues"] = actor_io_queues
        mem_args["learner_io_queue"] = learner_io_queue
        mem_args["io_learner_queue"] = io_learner_queue

        learner_args["learner_io_queue"] = learner_io_queue
        learner_args["io_learner_queue"] = io_learner_queue
        learner_args["learner_actor_queues"] = learner_actor_queues

    return actor_args, mem_args, learner_args, env_args, global_config, queues
