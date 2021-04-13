import os
import yaml
import glob
import torch
import numpy as np
from iniparser import Config
from distributed.model_util import choose_model, load_model
from distributed.eval_util import (
    calculate_theoretical_max_q_value,
    create_user_eval_state,
    run_evaluation_in_batches,
)
from surface_rl_decoder.surface_code_util import (
    SOLVED_EPISODE_REWARD,
    STATE_MULTIPLIER,
    SYNDROME_DIFF_REWARD,
    compute_intermediate_reward,
)
from surface_rl_decoder.surface_code import SurfaceCode
from distributed.util import select_actions
import matplotlib.pyplot as plt


def check_inter_reward(model, eval_device):
    surface_code = SurfaceCode()
    code_size = surface_code.code_size
    stack_depth = surface_code.stack_depth

    manual_episode = True

    if manual_episode:
        state, expected_actions, theoretical_q_value = create_user_eval_state(
            surface_code, 0, discount_factor_gamma=0.95
        )

        surface_code.state = state
        # surface_code.render()
    else:
        surface_code.reset()
        # surface_code.render()
        state = surface_code.state

    # states = torch.tensor(state[None, :, :, :], dtype=torch.float32).to(eval_device)
    # actions, q_values= select_actions(states, model, code_size)
    # action = actions[0]
    # next_state, reward, terminal, _, = surface_code.step(action, )
    # print(f"{action=}")
    print(f"{SYNDROME_DIFF_REWARD=}")
    print(f"{SOLVED_EPISODE_REWARD=}")
    manual_stack_depth = 8
    discount_inter_reward = 0.3
    gamma = 0.95
    print(f"{gamma=}")
    print(f"{discount_inter_reward=}")
    print("")

    # case: no syndrome
    state = np.zeros((manual_stack_depth, code_size + 1, code_size + 1))
    next_state = np.zeros((manual_stack_depth, code_size + 1, code_size + 1))
    next_state[:, 0, 1] = STATE_MULTIPLIER
    next_state[:, 1, 2] = STATE_MULTIPLIER

    inter_rew = compute_intermediate_reward(
        state, next_state, manual_stack_depth, discount_factor=discount_inter_reward
    )

    theo_q_value = calculate_theoretical_max_q_value(
        state, gamma=gamma, discount_inter_reward=discount_inter_reward
    )
    syndrome_depth = 0
    print(f"{syndrome_depth=}, {theo_q_value=}, {inter_rew=}")
    print("")

    for syndrome_depth in range(1, manual_stack_depth + 1):
        # syndrome_depth = 3
        state = np.zeros((manual_stack_depth, code_size + 1, code_size + 1))
        state[-syndrome_depth:, 0, 1] = STATE_MULTIPLIER
        state[-syndrome_depth:, 1, 2] = STATE_MULTIPLIER

        # print(f"{state[-1]=}")

        next_state = np.zeros((manual_stack_depth, code_size + 1, code_size + 1))
        next_state[:-syndrome_depth, 0, 1] = STATE_MULTIPLIER
        next_state[:-syndrome_depth, 1, 2] = STATE_MULTIPLIER

        inter_rew = compute_intermediate_reward(
            state, next_state, manual_stack_depth, discount_factor=discount_inter_reward
        )

        theo_q_value = calculate_theoretical_max_q_value(
            state, gamma=gamma, discount_inter_reward=discount_inter_reward
        )
        print(f"{syndrome_depth=}, {theo_q_value=}, {inter_rew=}")
        print("")


if __name__ == "__main__":
    cfg = Config()
    _config = cfg.scan(".", True).read()
    config = cfg.config_rendered.get("eval_config")
    eval_config = config.get("eval")
    eval_device = eval_config.get("device", "cpu")

    load_path = eval_config.get("load_model_path")

    for filename in glob.glob(load_path + "*"):
        if filename.endswith(".pt"):
            model_file = filename

        if "meta" in filename:
            meta_file = filename

    with open(meta_file) as meta_fp:
        metadata = yaml.load(meta_fp)

    network_config = metadata["network"]
    network_config["device"] = eval_device
    network_name = network_config["name"]

    # os.environ["CONFIG_ENV_SIZE"] = metadata["global"]["env"]["size"]
    # os.environ["CONFIG_ENV_STACK_DEPTH"] = metadata["global"]["env"]["stack_depth"]

    model = choose_model(network_name, network_config)
    model, *_ = load_model(model, old_model_path=model_file, model_device=eval_device)
    check_inter_reward(model, eval_device)
