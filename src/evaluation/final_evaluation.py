from time import time
import os
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
import glob
import torch
import numpy as np
from iniparser import Config
from distributed.model_util import choose_model, load_model
from distributed.eval_util import create_user_eval_state
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
    RESULT_KEY_EPISODE
)
from distributed.learner_util import log_evaluation_data, safe_append_in_dict, transform_list_dict
from surface_rl_decoder.surface_code_util import (
    STATE_MULTIPLIER,
    TERMINAL_ACTION,
    compute_intermediate_reward,
    compute_layer_diff,
    create_syndrome_output_stack,
)
from surface_rl_decoder.surface_code import SurfaceCode
from distributed.util import select_actions
import matplotlib.pyplot as plt


def count_spikes(arr):
    arr_length = len(arr)
    num_spikes = 0
    if arr_length < 3:
        return 0

    for i in range(1, arr_length - 1):
        if arr[i - 1] < arr[i] and arr[i + 1] < arr[i]:
            num_spikes += 1
        if arr[i - 1] > arr[i] and arr[i + 1] > arr[i]:
            num_spikes += 1

    return num_spikes / (arr_length - 2)


def count_spikes_np(arr, verbosity=0):
    arr_length = len(arr)

    if arr_length < 3:
        return 0

    first_derivative = np.diff(arr)
    sign_first_derivative = np.sign(first_derivative)
    nonzero_sign_first_derivative = sign_first_derivative[first_derivative != 0]
    second_derivative = np.diff(nonzero_sign_first_derivative)

    nonzero_second_derivative = second_derivative.nonzero()[0]

    spike_idx = nonzero_second_derivative + 1
    if verbosity:
        print(f"{sign_first_derivative=}")
        print(f"{nonzero_sign_first_derivative=}")
        print(f"{second_derivative=}")
        print(f"{spike_idx=}")
        print(f"{len(spike_idx)=}")

    return len(spike_idx) / (arr_length - 2)


def calc_net_summed_gain(arr):
    """
    Calculate for each detrimental step, how
    severe the action was on the array values.

    Can be used to look for moments in which the energy
    temporarily rises in one step.
    """
    arr_length = len(arr)
    net_gain = 0
    counter = 0
    if arr_length < 2:
        return 0

    for i in range(1, arr_length):
        if (diff := arr[i] - arr[i - 1]) > 0:
            net_gain += diff
            counter += 1
    if counter == 0:
        return 0

    return net_gain / counter


def calc_net_summed_gain_np(arr):
    arr_length = len(arr)
    if arr_length < 2:
        return 0

    diffs = np.diff(arr)
    positive_diffs = np.argwhere(diffs > 0)
    counter = len(positive_diffs)
    if counter == 0:
        return 0
    net_gain = np.sum(diffs[positive_diffs])

    return net_gain / counter


def calc_worsening(arr):
    """
    Look for moments which result in negative array entries
    and compute their severity.
    """

    arr_length = len(arr)
    avg_worsening = 0
    counter = 0
    for i in range(0, arr_length):
        if arr[i] < 0:
            avg_worsening += arr[i]
            counter += 1
    if counter == 0:
        return 0
    return avg_worsening / counter


def calc_worsening_np(arr):
    negative_entries = np.argwhere(arr < 0)
    worsening = np.sum(arr[negative_entries])
    counter = len(negative_entries)

    if counter == 0:
        return 0

    return worsening / counter


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
    rewards = []
    terminal_energies = []
    n_syndromes_created = []
    n_syndromes_annihilated = []
    # surface_code.render()
    while not terminal:
        step_counter += 1
        if step_counter > 10:
            break

        energy = np.sum(states[0, -1, :, :]) / STATE_MULTIPLIER
        energies.append(energy)
        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        actions, q_values = select_actions(
            torch_states, model, code_size, epsilon=epsilon
        )

        # plt.plot(q_values[0])
        # plt.show()
        assert np.all(
            state == surface_code.state
        ), f"{state[-1]}, {surface_code.state[-1]}"
        action = actions[0]
        (
            next_state,
            reward,
            terminal,
            _,
        ) = surface_code.step(actions[0])
        print(f"{action}")

        if action[-1] != TERMINAL_ACTION:
            assert np.all(next_state == surface_code.next_state)
            assert not np.all(state == next_state)
            assert np.all(surface_code.state == next_state)
        rewards.append(reward)
        # NOTE intermediate rewards seem to be badly designed

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
            print(f"{inter_rew=}")
            print(f"{energy=}")
            inter_rews.append(inter_rew)
            assert inter_rew == reward

        state = next_state
        states = state[None, :, :, :]

        print("")
        # surface_code.render()

    if len(terminal_energies) != 0:
        print(f"Terminal energy: {terminal_energies[0]}")
    else:
        print("Episode was never terminated by the agent.")

    syndromes_annihilated = np.sum(n_syndromes_annihilated)
    syndromes_created = np.sum(n_syndromes_created)
    syndromes_normalization = syndromes_annihilated + syndromes_created
    syndromes_created /= syndromes_normalization
    syndromes_annihilated /= syndromes_normalization

    print(f"Syndromes annihilated per step: {syndromes_annihilated}")
    print("")

    print(f"Final energy: {energies[-1]}")
    # TODO: is there a way to normalize energy difference?
    print(f"Net energy difference: {energies[-1] - energies[0]}")
    # TODO: need difference metric for energy fluctuation
    # need to punish up and down movement in energy, resulting from repeating one action all the time
    energy_spikes = count_spikes(energies)
    energy_spikes_np = count_spikes_np(energies)
    assert (
        energy_spikes == energy_spikes_np
    ), f"{energy_spikes=}, {energy_spikes_np=}, {energies=}"
    print(f"Energy spikes: {energy_spikes_np}")
    # net_sum_gain_energy = calc_net_summed_gain(energies)
    # net_sum_gain_energy_np = calc_net_summed_gain_np(energies)
    # assert net_sum_gain_energy == net_sum_gain_energy_np, f"{net_sum_gain_energy=}, {net_sum_gain_energy_np=}"
    # print(f"Net intermed. energy gain per step: {net_sum_gain_energy_np}")
    num_energy_raises = len(np.argwhere(np.diff(energies) > 0))
    print(f"Number of energy raises: {2 * num_energy_raises / len(energies)}")
    print("")

    inter_rews = np.array(inter_rews)
    inter_rew_spikes = count_spikes(inter_rews)
    inter_rew_spikes_np = count_spikes_np(inter_rews)
    assert (
        inter_rew_spikes == inter_rew_spikes_np
    ), f"{inter_rew_spikes=}, {inter_rew_spikes_np=}, {inter_rews=}"
    print(f"Intermediate reward spikes: {inter_rew_spikes_np}")
    # inter_rew_worsening = calc_worsening(np.array(inter_rews))
    # inter_rew_worsening_np = calc_worsening_np(np.array(inter_rews))
    # assert inter_rew_worsening == inter_rew_worsening_np, f"{inter_rew_worsening=}, {inter_rew_worsening_np=}"
    # print(f"Net intermediate reward loss per step: {inter_rew_worsening_np}")
    num_negative_inter_rewards = len(np.argwhere(inter_rews < 0))
    print(
        f"Number of negative intermediate rewards: {2 * num_negative_inter_rewards / len(inter_rews)}"
    )
    # TODO: think about good measures for intermediate rewards
    # what are the expectations
    # what can we learn from extreme inter rewards (like max and min)
    # maybe look at how often those extrema occur
    # in principle, we already cover the spikiness with the function above
    print(
        f"Average positive intermediate reward: {np.mean(inter_rews[inter_rews > 0])}"
    )
    # print(f"Max inter reward: {np.max(inter_rews)}")
    print(f"Min inter reward: {np.min(inter_rews)}")
    # print(f"Difference extreme inter rewards: {np.min(inter_rews) - np.max(inter_rews)}")

    if energy_spikes < 1 or inter_rew_spikes < 1 or syndromes_annihilated != 0.5:
        plt.plot(energies)
        plt.title("Energy")
        plt.show()

        plt.plot(inter_rews, label="inter. reward")
        plt.plot(
            rewards,
            label="reward",
        )
        plt.title("Intermediate Reward")
        plt.legend()
        plt.show()


# TODO: investigate the following scenario:
# qubits = np.zeros((stack_depth, code_size, code_size))
# qubits[:, 2, 1] = 1
# qubits[:, 1, 2] = 2

# then the first action
# qubits[:, 2, 2] = 3

# and then act with (2, 2, 2) repeatedly on the resulting state


if __name__ == "__main__":
    cfg = Config()
    _config = cfg.scan(".", True).read()
    config = cfg.config_rendered.get("eval_config")
    eval_config = config.get("eval")
    _env_config = cfg.config_rendered.get("config")
    env_config = _env_config.get("env")
    stack_depth = int(env_config.get("stack_depth"))
    code_size = int(env_config.get("size"))

    eval_device = eval_config.get("device", "cpu")

    load_path = eval_config.get("load_model_path")

    verbosity = 5
 
    summary_path = "testing"
    summary_date = "1"

    tb_path = os.path.join(summary_path, summary_date, "learner")
    tensorboard = SummaryWriter(tb_path) #TODO init tb

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

    # input_to_model = np.random.randint(0, 2, (1, stack_depth, code_size+1, code_size+1))
    # input_to_model = torch.tensor(input_to_model, dtype=torch.float32, device=eval_device)
    # tensorboard.add_graph(model, input_to_model=input_to_model, verbose=False)

    # TODO: investigate the following scenario:
    # surface_code = SurfaceCode()
    # stack_depth = surface_code.stack_depth
    # code_size = surface_code.code_size
    # qubits = np.zeros((stack_depth, code_size, code_size), dtype=np.uint8)
    # qubits[:, 1, 0] = 1
    # qubits[:, 2, 1] = 2

    # # then the first action
    # qubits[:, 2, 2] = 1
    # surface_code.qubits = qubits
    # state = create_syndrome_output_stack(qubits, surface_code.vertex_mask, surface_code.plaquette_mask)
    # surface_code.state = state
    # surface_code.render()

    # action = np.array([2, 1, 3], dtype=np.uint8)
    # surface_code.step(action)
    # surface_code.render()
    # and then act with (2, 2, 2) repeatedly on the resulting state

    # main_evaluation(model, eval_device)
    for t in range(5):
        # final_result_dict = {
        #     RESULT_KEY_EPISODE_AVG: {},
        #     RESULT_KEY_EPISODE_MEDIAN: {},
        #     RESULT_KEY_Q_VALUE_STATS: {},
        #     RESULT_KEY_AVG_INCREASING: {},
        #     RESULT_KEY_MEDIAN_INCREASING: {},
        #     RESULT_KEY_AVG_COUNTS: {},
        #     RESULT_KEY_MEDIAN_COUNTS: {},
        #     RESULT_KEY_AVG_RATES: {},
        #     RESULT_KEY_MEDIAN_RATES: {},
        # }
        final_result_dict = {
            RESULT_KEY_EPISODE: {},
            RESULT_KEY_Q_VALUE_STATS: {},
            RESULT_KEY_ENERGY: {},
            RESULT_KEY_COUNTS: {},
            RESULT_KEY_RATES: {},
        }
        p_error_list = [0.01, 0.02]
        for i_err_list, p_error in enumerate(p_error_list):

            eval_results, all_q_values = batch_evaluation(
                model,
                "",
                eval_device,
                num_of_random_episodes=4,
                num_of_user_episodes=4,
                max_num_of_steps=10,
                discount_intermediate_reward=0.3,
                p_err=p_error,
                p_msmt=0.0,
                verbosity=5,
            )
            # print(f"{eval_results.keys()=}")
            # TODO: is there a way to construct final_result_dict just from eval_results?
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
            tensorboard,
            tb_results,
            p_error_list,
            t,
            current_time_tb
        )

        if verbosity >= 4:
            for p_err in p_error_list:
                tensorboard.add_histogram(
                    f"network/q_values, p_error {p_err}",
                    all_q_values[RESULT_KEY_HISTOGRAM_Q_VALUES],
                    t,
                    walltime=current_time_tb,
                )