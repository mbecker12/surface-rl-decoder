import os
import yaml
import glob
import torch
import numpy as np
from iniparser import Config
from distributed.model_util import choose_model, load_model
from distributed.eval_util import create_user_eval_state, run_evaluation_in_batches
from surface_rl_decoder.surface_code_util import (
    STATE_MULTIPLIER,
    compute_intermediate_reward,
)
from surface_rl_decoder.surface_code import SurfaceCode
from distributed.util import select_actions
import matplotlib.pyplot as plt


def main_evaluation(model, device, epsilon=0.0):
    surface_code = SurfaceCode()
    code_size = surface_code.code_size
    state, expected_actions, theoretical_q_value = create_user_eval_state(
        surface_code, 0, discount_factor_gamma=0.95
    )

    surface_code.reset()
    stack_depth = surface_code.stack_depth
    code_size = surface_code.code_size
    # surface_code.state = state
    # surface_code.state = np.ones((stack_depth, code_size+1, code_size+1))
    state = surface_code.state
    p_err = surface_code.p_error
    p_msmt = surface_code.p_msmt
    print(f"{p_err=}, {p_msmt=}")

    states = state[None, :, :, :]

    initial_x_errors = np.argwhere(surface_code.qubits[-1, :, :] == 1)
    initial_y_errors = np.argwhere(surface_code.qubits[-1, :, :] == 2)
    initial_z_errors = np.argwhere(surface_code.qubits[-1, :, :] == 3)
    print(f"{initial_x_errors=}")
    print(f"{initial_y_errors=}")
    print(f"{initial_z_errors=}")
    terminal = False
    step_counter = 0
    energies = []
    inter_rews = []
    surface_code.render()
    while not terminal:
        step_counter += 1
        if step_counter >= 10:
            break

        energy = np.sum(states[0, -1, :, :]) / STATE_MULTIPLIER
        energies.append(energy)
        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        actions, q_values = select_actions(
            torch_states, model, code_size, epsilon=epsilon
        )

        # plt.plot(q_values[0])
        # plt.show()
        action = actions[0]
        (
            next_state,
            reward,
            terminal,
            _,
        ) = surface_code.step(actions[0])

        # NOTE intermediate rewards seem to be badly designed
        inter_rew = compute_intermediate_reward(state, next_state, stack_depth)
        inter_rews.append(inter_rew)
        states = next_state[None, :, :, :]
        print(f"{action}")
        # surface_code.render()

    plt.plot(energies)
    plt.title("Energy")
    plt.show()

    plt.plot(inter_rews)
    plt.title("Intermediate Reward")
    plt.show()


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
    main_evaluation(model, eval_device)
