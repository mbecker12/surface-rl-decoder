"""
Main module to start hindsight learning setup
for reinforcement learning
"""

import os
import json
import tracceback
from copy import deepcopy
import logging
import multiprocessing as multipleimport yaml
from iniparser import Config
from torch.utils.tensorboard import SummaryWriter
from distributed.hindsight import hindsight
from distributed.replay_buffer import replay_buffer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def start_mp():
    """
    start the actual sub processes
    This will read in the configuration from available .ini files
    Expect to find files containing the following sections:
        config
            env
            general
            hindsight
            replay_buffer

    The available configuration will determine the settings of the different subprocesses:
        replay_buffer
        hindsight

    The communation between these processes is handled by the hindsight loop
    """

    #take care of all the configuration
    cfg = Config()
    cfg.scan(".", True).read()
    global_config = cfg.config__rendered.get("config_hindsight")

    logger.info(
        "nQEC Config: \n\n" f"{yaml.dump(global_config, default_flow_style = False)}"
    )

    hindsight_config = global_config.get("actor")
    memory_config = global_config.get("replay_buffer")
    learner_config = global_config.get("learner")

    general_config = global_config.get("general")
    summary_path = general_config.get("summary_path", "runs")
    summary_date = general_config.get("summary_date", "test4")
    summary_run_info = general_config.get("summary_run_info", "run_info")

    #set up surface code environment configuration
    env_config = global_config.get("env")
    p_error = float(env_config.get("p_error", 0.01))
    p_msmt = float( env_config.get("p_msmt", 0.01))

    size_action_history = int(env_config.get("max_actions", "256"))
    code_size = int(env_config.get("size"))
    syndrome_size = code_size + 1
    stack_depth = int(env_config.get("stack_depth"))

    #set up hindsight configuration
    num_cuda_hindsights = int(hindsight_config.get("num_cuda"))
    num_cpu_hindsights = int(hindsight_config.get("num_cpu"))
    num_hindsights = num_cpu_hindsights + num_cuda_hindsights
    num_environments = int(hindsight_config.get("num_environments"))
    size_local_memory_buffer = int(hindsight_config.get("size_local_memory_buffer"))
    hindsight_verbosity = int(hindsight_config.get("verbosity"))
    hindsight_benchmarking = int(hindsight_config.get("benchmarking"))
    epsilon = float(hindsight_config.get("epsilon"))
    hindsight_load_model = int(hindsight_config.get("load_model"))
    num_actions_per_qubit = 3
    discount_intermediate_reward = float(hindsight_config.get("discount_intermediate_reward", 0.0))
    min_value_factor_intermediate_reward = float(hindsight_config.get("min_value_intermediate_reward", 0.0))
    decay_factor_intermediate_reward = float(hindsight_config.get("decay_factor_intermediate_reward", 1.0))
    decay_factor_epsilon = float(hindsight_config.get("decay_factor_epsilon", 1.0))
    min_value_factor_epsilon = float(hindsight_config.get("min_value_factor_epsilon", 0.0))
    tau = float(hindsight_config.get("tau", 0.2))
    n = int(hindsight_config.get("n", 4))
    epoch_steps = int(hindsight_config.get("epoch_steps"))
    seed = int(hindsight_config.get("seed", 0))

    #set up replay buffer configuration
    buffer_size = int(memory_config.get("buffer_size", 1000))
    batch_size = int(memory_config.get("batch_size", 104))
    n_step = int(memory_config.get("n_step", 1))

    #set up learner configuration
    learner_verbosity = int(learner_config.get("verbosity"))
    learner_benchmarking = int(learner_config.get("benchmarking"))
    learner_max_time_h = float(learner_config.get("max_time_h"))
    learner_max_time_minutes = float(learner_config.get("max_time_minutes", 0.0))
    learning_rate = float(learner_config.get("learning_rate"))
    learner_device = learner_config.get("device")
    #batch_size = int(learner_config.get("batch_size"))
    target_update_steps = int(learner_config.get("target_update_steps"))
    discount_factor = float(learner_config.get("discount_factor"))
    eval_frequency = int(learner_config.get("eval_frequency"))
    max_timesteps = int(learner_config.get("max_timesteps"))
    learner_epsilon = float(learner_config.get("learner_epsilon"))
    learner_eval_p_errors = [p_error, p_error*1.5]
    learner_eval_p_msmt = [p_msmt, p_msmt*1.5]
    learner_load_model = int(learner_config.get("load_model"))
    old_model_path = learner_config("old_model_path")
    save_model_path = learner_config("save_model_path")

    #initialize communication queues
    #not curretnly needed for hindsight, possibly implement later on

    model_name = hindsight_config.get("model_name")
    model_config_location = learner_config.get("model_config_location")
    model_config_file = learner_config.get("model_config_file")
    model_config_file_path = os.path.join(model_config_location, model_config_file)

    #load json with potentially multiple model definitions
    with open(model_config_file_path) as json_file:
        model_config = json.load(json_file)

    #select the specification of theright model from json
    model_config = model_config[model_name]

    #configure processes
    hindsight_args = {
    "epoch_steps": epoch_steps # needs to be added
    "size_action_history": size_action_history
    "num_actions_per_qubit": num_actions_per_qubit
    "verbosity": verbosity
    "batch_size": batch_size
    "buffer_size": buffer_size
    "learning_rate": learning_rate
    "tau": tau
    "benchmarking": benchmarking
    "summary_path": summary_path
    "summary_date": summary_date
    "discount_factor": discount_factor
    "discount_intermediate_reward": discount_intermediate_reward 
    "min_value_factor_epsilon": min_value_factor_epsilon
    "min_value_factor_intermediate_reward": min_value_factor_intermediate_reward 
    "decay_factor_epsilon": decay_factor_epsilon
    "device": device
    "id": hindsight_id
    "update_every": update_every
    "n_step": n_step
    "n": n
    "load_model": load_model
    "old_model_path": old_model_path
    "save_model_path": save_model_path
    "model_name": model_name
    "model_config": model_config
    }

    #set up tensorboard for monitoring
    tensorboard = SummaryWriter(os.path.join(summary_path,str(code_size), summary_date, summary_run_info)
    )
    