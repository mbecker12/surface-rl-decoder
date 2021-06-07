import json
from analysis.analyze_general_training import TrainingRun
from copy import deepcopy
import os
from typing import Dict
import numpy as np
import torch
import yaml
from agents.base_agent import BaseAgent
from distributed.environment_set import EnvironmentSet
from distributed.util import q_value_index_to_action, select_actions
from distributed.model_util import (
    choose_model,
    choose_old_model,
    extend_model_config,
    load_model,
)
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION, check_final_state

CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "threshold_networks"
CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "threshold_networks"
BASE_MODEL_CONFIG_PATH = "src/config/model_spec/old_conv_agents.json"
BASE_MODEL_PATH = "remote_networks/5/65280/simple_conv_5_65280.pt"


def analyze_succesful_episodes(
    model: BaseAgent,
    environment_def,
    total_n_episodes=128,
    epsilon=0.0,
    max_num_of_steps=32,
    discount_factor_gamma=0.95,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.3,
    punish_repeating_actions=0,
    p_err=0.0,
    p_msmt=0.0,
    verbosity=0,
    rl_type="q",
    code_size=None,
    stack_depth=None,
    device="cpu",
) -> Dict:

    model.eval()

    if code_size is not None:
        os.environ["CONFIG_ENV_SIZE"] = str(code_size)
    if stack_depth is not None:
        os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)

    if environment_def is None or environment_def == "":
        environment_def = SurfaceCode(code_size=code_size, stack_depth=stack_depth)

    env_set = EnvironmentSet(environment_def, total_n_episodes)
    code_size = env_set.code_size
    stack_depth = env_set.stack_depth
    states = env_set.reset_all(
        np.repeat(p_err, total_n_episodes), np.repeat(p_msmt, total_n_episodes)
    )

    global_episode_steps = 0

    n_ground_states = 0
    n_valid_episodes = 0
    n_valid_ground_states = 0
    n_valid_non_trivial_loops = 0
    n_ep_w_syndromes = 0
    n_ep_w_loops = 0
    n_too_long = 0
    n_too_long_w_loops = 0
    n_too_long_w_syndromes = 0

    ignore_episodes = set()
    n_steps = np.zeros(total_n_episodes)
    while global_episode_steps <= max_num_of_steps:
        global_episode_steps += 1
        for i in range(total_n_episodes):
            if i not in ignore_episodes:
                n_steps[i] += 1

        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        if "q" in rl_type.lower():
            actions, tmp_q_values = select_actions(
                torch_states, model, code_size, epsilon=epsilon
            )
        elif "ppo" in rl_type.lower():
            if epsilon == 1:
                actions = model.select_greedy_action_ppo(
                    torch_states, return_logits=False, return_values=False
                )
            else:
                actions = model.select_action_ppo(
                    torch_states, return_logits=False, return_values=False
                )

            actions = np.array(
                [q_value_index_to_action(action, code_size) for action in actions]
            )

        next_states, rewards, terminals, _ = env_set.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )

        non_terminal_episodes = np.where(actions[:, -1] != TERMINAL_ACTION)[0]

        if np.any(terminals):
            indices = np.argwhere(terminals).flatten()
            for i in indices:
                if i in ignore_episodes:
                    continue
                ignore_episodes.add(i)

                _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                    env_set.environments[i].actual_errors,
                    env_set.environments[i].actions,
                    env_set.environments[i].vertex_mask,
                    env_set.environments[i].plaquette_mask,
                )

                if env_set.environments[i].current_action_index >= max_num_of_steps - 1:
                    print("line 110: failed episode; too many steps")
                    n_too_long += 1
                    if n_syndromes > 0:
                        n_too_long_w_syndromes += 1
                    if n_loops > 0:
                        n_too_long_w_loops += 1

                if n_syndromes == 0:
                    n_valid_episodes += 1
                    if _ground_state:
                        n_valid_ground_states += 1
                    else:
                        n_valid_non_trivial_loops += 1

                if _ground_state:
                    n_ground_states += 1
                if n_syndromes > 0:
                    n_ep_w_syndromes += 1
                if int(n_loops) > 0:
                    n_ep_w_loops += 1
                    if _ground_state:
                        print("Something's wrong, I can feel it.")

        if np.all(terminals):
            break

        states = next_states
        env_set.states = deepcopy(states)

    for i in range(total_n_episodes):
        if i in ignore_episodes:
            continue

        _, _ground_state, (n_syndromes, n_loops) = check_final_state(
            env_set.environments[i].actual_errors,
            env_set.environments[i].actions,
            env_set.environments[i].vertex_mask,
            env_set.environments[i].plaquette_mask,
        )

        if env_set.environments[i].current_action_index >= max_num_of_steps - 1:
            print("line 110: failed episode; too many steps")
            n_too_long += 1
            if n_syndromes > 0:
                n_too_long_w_syndromes += 1
            if n_loops > 0:
                n_too_long_w_loops += 1

        if n_syndromes == 0:
            n_valid_episodes += 1
            if _ground_state:
                n_valid_ground_states += 1
            else:
                n_valid_non_trivial_loops += 1

        if _ground_state:
            n_ground_states += 1
        if n_syndromes > 0:
            n_ep_w_syndromes += 1
        if int(n_loops) > 0:
            n_ep_w_loops += 1
            if _ground_state:
                print("Something's wrong, I can feel it.")

    result_dict = {
        "total_n_episodes": total_n_episodes,
        "n_ground_states": n_ground_states,
        "n_valid_episodes": n_valid_episodes,
        "n_valid_ground_states": n_valid_ground_states,
        "n_valid_non_trivial_loops": n_valid_non_trivial_loops,
        "n_ep_w_syndromes": n_ep_w_syndromes,
        "n_ep_w_loops": n_ep_w_loops,
        "n_too_long": n_too_long,
        "n_too_long_w_loops": n_too_long_w_loops,
        "n_too_long_w_syndromes": n_too_long_w_syndromes,
        "n_steps_arr": n_steps,
    }
    return result_dict


def provide_default_ppo_metadata(
    code_size,
    stack_depth,
):
    metadata = {
        "code_size": code_size,
        "channel_list": [
            32,
            64,
            32,
            16,
            8,
        ],
        "device": "cuda",
        "input_channels": 1,
        "kernel_depth": 3,
        "kernel_size": 3,
        "model_name": "conv3d",
        "name": "conv3d",
        "network_size": "slim",
        "neuron_list": [512, 256],
        "num_actions_per_qubit": 3,
        "padding_size": 1,
        "split_input_toggle": 0,
        "stack_depth": stack_depth,
        "syndrome_size": 6,
    }
    return metadata


def load_analysis_model(
    run: TrainingRun, local_network_path=LOCAL_NETWORK_PATH, eval_device="cpu"
) -> BaseAgent:
    os.environ["CONFIG_ENV_SIZE"] = str(run.code_size)
    os.environ["CONFIG_ENV_STACK_DEPTH"] = str(run.stack_depth)
    load_path = f"{local_network_path}/{run.code_size}/{run.job_id}"
    model_config_path = load_path + f"/{run.model_name}_{run.code_size}_meta.yaml"
    old_model_path = load_path + f"/{run.model_name}_{run.code_size}_{run.job_id}.pt"

    if run.rl_type == "ppo" and not os.path.exists(model_config_path):
        model_config = provide_default_ppo_metadata(run.code_size, run.stack_depth)
    else:
        with open(model_config_path, "r") as yaml_file:
            general_config = yaml.load(yaml_file)
            model_config = general_config["network"]

    model_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    print(f"Load Model for {run.job_id}")
    if int(run.job_id) < 70000:
        model = choose_old_model(run.model_name, model_config)
    else:
        base_model_config = None
        if run.transfer_learning:
            with open(BASE_MODEL_CONFIG_PATH, "r") as base_file:
                base_model_config = json.load(base_file)["simple_conv"]

            base_model_config = extend_model_config(
                base_model_config, run.code_size + 1, run.stack_depth
            )
            base_model_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            model_config["rl_type"] = run.rl_type
            model = choose_model(
                run.model_name,
                model_config,
                model_config_base=base_model_config,
                model_path_base=BASE_MODEL_PATH,
                transfer_learning=run.transfer_learning,
            )
        else:
            model = choose_model(
                run.model_name,
                model_config,
                model_config_base=base_model_config,
                model_path_base=BASE_MODEL_PATH,
                transfer_learning=run.transfer_learning,
            )

    model, _, _ = load_model(model, old_model_path, model_device=eval_device)

    return model
