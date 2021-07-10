"""
A collection of different evaluation routines.
Meant to be altered by the developer by changing
the truth value of different if statements to perform different tests.
"""
from time import time
import os
import sys
import glob
import yaml
import argparse

# pylint: disable=not-callable
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from iniparser import Config
import matplotlib.pyplot as plt
from distributed.model_util import choose_model, choose_old_model, load_model
from distributed.learner_util import (
    log_evaluation_data,
    safe_append_in_dict,
    transform_list_dict,
)
from distributed.util import COORDINATE_SHIFTS, LOCAL_DELTAS, format_torch, q_value_index_to_action, select_actions, select_actions_value_network
from surface_rl_decoder.surface_code_util import (
    STATE_MULTIPLIER,
    TERMINAL_ACTION,
    check_final_state,
    compute_intermediate_reward,
    compute_layer_diff,
    create_syndrome_output_stack,
)
from surface_rl_decoder.surface_code import SurfaceCode
from evaluation.eval_util import count_spikes, create_user_eval_state
from surface_rl_decoder.syndrome_masks import get_plaquette_mask, get_vertex_mask

sys.path.append("/home/marvin/Projects/surface-rl-decoder")
sys.path.append("/home/marvin/Projects/surface-rl-decoder/analysis")
from analysis.analysis_util import provide_default_ppo_metadata

# pylint: disable=unused-import
from evaluation.batch_evaluation import (
    RESULT_KEY_ENERGY,
    batch_evaluation,
    RESULT_KEY_EPISODE_AVG,
    RESULT_KEY_EPISODE_MEDIAN,
    RESULT_KEY_STEP,
    RESULT_KEY_Q_VALUE_STATS,
    RESULT_KEY_HISTOGRAM_Q_VALUES,
    RESULT_KEY_AVG_INCREASING,
    RESULT_KEY_MEDIAN_INCREASING,
    RESULT_KEY_AVG_COUNTS,
    RESULT_KEY_MEDIAN_COUNTS,
    RESULT_KEY_AVG_RATES,
    RESULT_KEY_MEDIAN_RATES,
    RESULT_KEY_RATES,
    RESULT_KEY_COUNTS,
    RESULT_KEY_INCREASING,
    RESULT_KEY_EPISODE,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-locals, too-many-statements
def main_evaluation(
    model,
    device,
    epsilon=0.0,
    code_size=None,
    stack_depth=None,
    block=False,
    verbosity=0,
    rl_type="q",
):
    """
    The main program to be executed.
    Visualizes the surface code before and after
    the agent has done its work.
    Gives additional info about the evolution of energy and
    intermediate rewards.
    """
    # pylint: disable=redefined-outer-name
    surface_code = SurfaceCode(code_size=code_size, stack_depth=stack_depth)
    code_size = surface_code.code_size
    # state, _, _ = create_user_eval_state(surface_code, 0, discount_factor_gamma=0.95)

    surface_code.reset()
    stack_depth = surface_code.stack_depth
    code_size = surface_code.code_size

    state = surface_code.state
    p_err = surface_code.p_error
    p_msmt = surface_code.p_msmt
    if verbosity:
        print(f"{p_err=}, {p_msmt=}")

    states = state[None, :, :, :]

    initial_x_errors = np.argwhere(surface_code.qubits[-1, :, :] == 1)
    initial_y_errors = np.argwhere(surface_code.qubits[-1, :, :] == 2)
    initial_z_errors = np.argwhere(surface_code.qubits[-1, :, :] == 3)
    if verbosity:
        print(f"{initial_x_errors=}")
        print(f"{initial_y_errors=}")
        print(f"{initial_z_errors=}")
    terminal = False
    step_counter = 0
    energies = []
    inter_rews = []
    rewards = []
    terminal_energies = []
    n_syndromes_created = []
    n_syndromes_annihilated = []
    surface_code.render(block=block)
    action_history = []
    while not terminal:
        step_counter += 1
        if step_counter > 10:
            break

        energy = np.sum(states[0, -1, :, :]) / STATE_MULTIPLIER
        energies.append(energy)
        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        if rl_type == "q":
            actions, _ = select_actions(torch_states, model, code_size, epsilon=epsilon)
        elif rl_type == "ppo":
            actions = model.select_action_ppo(torch_states)
        elif rl_type == "v":
            plaquette_mask = get_plaquette_mask(code_size)
            vertex_mask = get_vertex_mask(code_size)
            combined_mask = np.logical_or(plaquette_mask, vertex_mask)
            combined_mask = format_torch(combined_mask)
            (
                actions, selected_values, optimal_actions, optimal_values
            ) = select_actions_value_network(
                torch_states,
                model,
                code_size,
                stack_depth,
                combined_mask,
                COORDINATE_SHIFTS,
                LOCAL_DELTAS,
                device
            )
            if verbosity:
                print(f"{selected_values[0]}")

        assert np.all(
            state == surface_code.state
        ), f"{state[-1]}, {surface_code.state[-1]}"

        if rl_type in ("q", "v"):
            action = actions[0]
        elif rl_type == "ppo":
            actions = actions.numpy()
            action = q_value_index_to_action(actions, code_size)

        action_history.append(action)
        (
            next_state,
            reward,
            terminal,
            _,
        ) = surface_code.step(action)
        if verbosity:
            print(f"{action}")

        if action[-1] != TERMINAL_ACTION:
            assert np.all(next_state == surface_code.next_state)
            assert not np.all(state == next_state)
            assert np.all(surface_code.state == next_state)
        rewards.append(reward)

        # assert not np.all(surface_code.next_state == next_state)
        if terminal:
            inter_rews.append(0)
            terminal_energies.append(energy)
            n_syndromes_annihilated.append(0)
            n_syndromes_created.append(0)
        else:
            diffs = compute_layer_diff(state, next_state, stack_depth)
            n_syndromes_annihilated.append(np.sum(diffs > 0))
            n_syndromes_created.append(np.sum(diffs < 0))
            inter_rew = compute_intermediate_reward(state, next_state, stack_depth)
            if verbosity:
                print(f"{inter_rew=}")
            if verbosity:
                print(f"{energy=}")
            inter_rews.append(inter_rew)
            if verbosity:
                print(f"{np.min(state)=}, {np.max(state)=}")
            assert inter_rew == reward, f"{inter_rew=}, {reward=}"

        state = next_state
        states = state[None, :, :, :]

        if verbosity:
            print("")
        # surface_code.render()

    final_state, is_ground_state, (n_syndromes, n_loops) = check_final_state(
        surface_code.actual_errors,
        action_history,
        surface_code.vertex_mask,
        surface_code.plaquette_mask,
    )

    (
        next_state,
        reward,
        terminal,
        _,
    ) = surface_code.step(action)
    if reward != rewards[-1]:
        rewards.append(reward)

    if verbosity:
        if len(terminal_energies) != 0:
            print(f"Terminal energy: {terminal_energies[0]}")
        else:
            print("Episode was never terminated by the agent.")

    syndromes_annihilated = np.sum(n_syndromes_annihilated)
    syndromes_created = np.sum(n_syndromes_created)
    syndromes_normalization = syndromes_annihilated + syndromes_created
    syndromes_created /= syndromes_normalization
    syndromes_annihilated /= syndromes_normalization

    if verbosity:
        print(f"Syndromes annihilated per step: {syndromes_annihilated}")
        print("")

        print(f"Final energy: {energies[-1]}")
        print(f"Net energy difference: {energies[-1] - energies[0]}")
    # need to punish up and down movement in energy,
    # resulting from repeating one action all the time
    energy_spikes = count_spikes(energies)

    num_energy_raises = len(np.argwhere(np.diff(energies) > 0))
    if verbosity:
        print(f"Number of energy raises: {2 * num_energy_raises / len(energies)}")
        print("")

    inter_rews = np.array(inter_rews)
    inter_rew_spikes = count_spikes(inter_rews)
    if verbosity:
        print(f"Intermediate reward spikes: {inter_rew_spikes}")

    num_negative_inter_rewards = len(np.argwhere(inter_rews < 0))
    if verbosity:
        print(
            "Number of negative intermediate rewards: "
            f"{2 * num_negative_inter_rewards / len(inter_rews)}"
        )
        print(
            f"Average positive intermediate reward: {np.mean(inter_rews[inter_rews > 0])}"
        )
        print(f"Min inter reward: {np.min(inter_rews)}")

        print(f"Final reward: {rewards[-1]}")

    if energy_spikes < 1 or inter_rew_spikes < 1 or syndromes_annihilated != 0.5:
        plt.plot(energies)
        plt.title("Energy")
        plt.show(block=block)

        plt.plot(inter_rews, label="inter. reward")
        plt.plot(
            rewards,
            label="reward",
        )
        plt.title("Rewards")
        plt.legend()
        plt.show(block=block)

        surface_code.render(block=block)

    return is_ground_state, n_syndromes, n_loops


if __name__ == "__main__":
    # Contains different tests which can be switched on or off by different
    # if statements (if True/False).
    # pylint: disable=using-constant-test
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--code_size", metavar="d", type=str)
    parser.add_argument("-H", "--stack_depth", metavar="H", type=str)
    args = parser.parse_args()

    cfg = Config()
    _config = cfg.scan(".", True).read()
    config = cfg.config_rendered.get("eval_config")
    eval_config = config.get("eval")
    _env_config = cfg.config_rendered.get("config")
    env_config = _env_config.get("env")
    stack_depth = int(env_config.get("stack_depth"))
    code_size = int(env_config.get("size"))

    if args.code_size is not None:
        code_size = int(args.code_size)
    if args.stack_depth is not None:
        stack_depth = int(args.stack_depth)

    eval_device = eval_config.get("device", "cpu")

    load_path = eval_config.get("load_model_path")

    verbosity = 5

    summary_path = "testing"
    summary_date = "1"

    tb_path = os.path.join(summary_path, summary_date, "learner")
    tensorboard = SummaryWriter(tb_path)

    meta_file = None
    rl_type = eval_config.get("rl_type")

    for filename in glob.glob(load_path + "*"):
        if filename.endswith(".pt"):
            model_file = filename

        if "meta" in filename:
            meta_file = filename

    if meta_file is not None:
        with open(meta_file) as meta_fp:
            metadata = yaml.load(meta_fp)

        network_config = metadata["network"]
    else:
        network_config = provide_default_ppo_metadata(code_size, stack_depth)
        rl_type = "ppo"

    network_config["rl_type"] = rl_type
    network_config["device"] = eval_device
    network_name = network_config["name"]

    model = choose_model(network_name, network_config, transfer_learning=0)
    # model = choose_old_model(network_name, network_config)
    model, *_ = load_model(model, old_model_path=model_file, model_device=eval_device)

    # check model layers, conv filters
    if False:
        layer_one = model.parameters().__next__().detach().numpy()
        print(f"{layer_one.shape=}")
        channels, _, z, x, y = layer_one.shape

        fig, ax = plt.subplots(channels, z)
        for c in range(channels):
            for k in range(z):
                ax[c, k].imshow(layer_one[c, 0, k, :, :], cmap="Greys")

        plt.show()

    # input_to_model = np.random.randint(0, 2, (1, stack_depth, code_size+1, code_size+1))
    # input_to_model = torch.tensor(input_to_model, dtype=torch.float32, device=eval_device)
    # tensorboard.add_graph(model, input_to_model=input_to_model, verbose=False)

    # investigate a specific scenario
    if False:
        surface_code = SurfaceCode()
        stack_depth = surface_code.stack_depth
        code_size = surface_code.code_size
        qubits = np.zeros((stack_depth, code_size, code_size), dtype=np.uint8)
        qubits[:, 1, 0] = 1
        qubits[:, 2, 1] = 2

        # then the first action
        qubits[:, 2, 2] = 1
        surface_code.qubits = qubits
        state = create_syndrome_output_stack(
            qubits, surface_code.vertex_mask, surface_code.plaquette_mask
        )
        surface_code.state = state
        surface_code.render()

        action = np.array([2, 1, 3], dtype=np.uint8)
        # surface_code.step(action)
        # surface_code.render()
        # and then act with (2, 2, 2) repeatedly on the resulting state

    # perform the main evaluation
    if True:
        n_episodes = 250
        n_ground_states = 0
        n_valid_ground_states = 0
        n_ep_w_syndromes = 0
        n_ep_w_loops = 0
        n_valid_episodes = 0
        n_valid_non_trivial_loops = 0
        for i in range(n_episodes):
            sys.stdout.write(f"\r{i+1:05d} / {n_episodes:05d}")
            is_ground_state, n_syndromes, n_loops = main_evaluation(
                model, eval_device, code_size=code_size, stack_depth=stack_depth, rl_type=rl_type
            )
            if n_syndromes == 0:
                n_valid_episodes += 1
                if is_ground_state:
                    n_valid_ground_states += 1
                else:
                    n_valid_non_trivial_loops += 1

            if is_ground_state:
                n_ground_states += 1
            if n_syndromes > 0:
                n_ep_w_syndromes += 1
            if int(n_loops) > 0:
                n_ep_w_loops += 1
                if is_ground_state:
                    print("Something's wrong, I can feel it.")
        print("\n")

        print(
            f"\n{n_episodes=}, {n_ground_states=}, "
            f"episodes with syndromes={n_ep_w_syndromes} "
            f"episodes with non-trivial loops={n_ep_w_loops}\n"
            f"\nValid episodes: {n_valid_episodes}, "
            f"Valid ground state episodes: {n_valid_ground_states}, "
            f"Fraction of valid ground states: {n_valid_ground_states / n_valid_episodes:.4f}, "
            f"Valid Fail Rate: {1.0 - n_valid_ground_states / n_valid_episodes:.4f}, "
            f"Valid episodes w/ non-trivial loops: {n_valid_non_trivial_loops}"
        )

    # perform visual evaluation
    if False:
        main_evaluation(model, eval_device, block=True, rl_type=rl_type, verbosity=1)

    # test integration with evaluation routine in the real program
    if False:
        for t in range(1):
            final_result_dict = {
                RESULT_KEY_EPISODE: {},
                RESULT_KEY_Q_VALUE_STATS: {},
                RESULT_KEY_ENERGY: {},
                RESULT_KEY_COUNTS: {},
                RESULT_KEY_RATES: {},
            }
            p_error_list = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
            for i_err_list, p_error in enumerate(p_error_list):

                eval_results, all_q_values = batch_evaluation(
                    model,
                    "",
                    eval_device,
                    num_of_random_episodes=64,
                    num_of_user_episodes=0,
                    max_num_of_steps=40,
                    discount_intermediate_reward=0.3,
                    p_err=p_error,
                    p_msmt=0.0,
                    verbosity=5,
                    code_size=code_size,
                    stack_depth=stack_depth,
                )

                for category_name, category in eval_results.items():
                    for key, val in category.items():
                        final_result_dict[category_name] = safe_append_in_dict(
                            final_result_dict[category_name], key, val
                        )

            tb_results = {}
            for key, values in final_result_dict.items():
                tb_results[key] = transform_list_dict(values)

            current_time_tb = time()
            # print(f"{tb_results=}")
            log_evaluation_data(
                tensorboard, tb_results, p_error_list, t, current_time_tb
            )

            if verbosity >= 4:
                for p_err in p_error_list:
                    tensorboard.add_histogram(
                        f"network/q_values, p_error {p_err}",
                        all_q_values[RESULT_KEY_HISTOGRAM_Q_VALUES],
                        t,
                        walltime=current_time_tb,
                    )
            print(f"{tb_results=}")
            # print(f"{yaml.dump(tb_results, default_flow_style=False)}")
