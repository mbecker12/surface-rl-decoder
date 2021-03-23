import numpy as np
from copy import deepcopy
import torch
from torch import from_numpy
import gym
import logging
from distributed.util import action_to_q_value_index, incremental_mean, select_actions
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import (
    TERMINAL_ACTION,
    check_final_state,
    compute_layer_diff,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


def evaluate(
    model,
    env,
    device,
    p_error_list,
    p_msmt_list,
    num_of_episodes=10,
    num_actions=3,
    epsilon=0.0,
    num_of_steps=50,
    plot_one_episode=True,
):
    """
    Evaluate the current policy.
    """

    model.eval()

    # TODO: publish gym environment
    # TODO: make environment configurable
    # env = gym.make(env, config=conf)

    env = SurfaceCode()
    system_size = env.system_size
    stack_depth = env.stack_depth
    n_qubits_total = stack_depth * system_size * system_size
    assert (
        system_size % 2 == 1
    ), "System size (i.e. number of qubits) needs to be an odd number."

    # initialize arrays for scoring metrics
    ground_state_list = np.zeros(len(p_error_list))
    fully_corrected_list = np.zeros(len(p_error_list))
    syndromes_annihilated_list = np.zeros(len(p_error_list))
    syndromes_created_list = np.zeros(len(p_error_list))
    remaining_syndromes_list = np.zeros(len(p_error_list))
    logical_errors_list = np.zeros(len(p_error_list))
    average_number_of_steps_list = np.zeros(len(p_error_list))
    mean_q_list = np.zeros(len(p_error_list))
    failed_syndromes = []

    for i_err_list, p_error in enumerate(p_error_list):
        p_msmt = p_msmt_list[i_err_list]
        ground_state = np.ones(num_of_episodes, dtype=bool)
        fully_corrected = np.zeros(num_of_episodes, dtype=bool)
        syndromes_annihilated = np.zeros(num_of_episodes, dtype=int)
        syndromes_created = np.zeros(num_of_episodes, dtype=int)
        remaining_syndromes = np.zeros(num_of_episodes, dtype=int)
        logical_errors = np.zeros(num_of_episodes, dtype=int)
        mean_steps_per_p_error = 0
        mean_q_value_per_p_error = 0
        steps_counter = 0

        # TODO: maybe one can evaluate in batches
        for j_episode in range(num_of_episodes):
            steps_counter = 0
            logger.debug(f"{p_error=}, episode: {j_episode}")
            num_steps_per_episode = 0
            previous_action = (0, 0, 0)
            chosen_previous_action = 0
            terminal_state = 0
            terminal = False

            state = env.reset(p_error=p_error, p_msmt=p_msmt)
            torch_state = torch.tensor(state, dtype=torch.float32).to(device)
            # if len(state.shape) < 4:
            torch_state = torch.unsqueeze(torch_state, 0)

            energy_surface = []
            experimental_q_values = []
            while not terminal and num_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_steps_per_episode += 1

                actions, q_values = select_actions(
                    torch_state, model, system_size, epsilon=epsilon
                )

                q_value_index = action_to_q_value_index(actions[0], system_size)
                q_value = q_values[0, q_value_index]
                experimental_q_values.append(q_value)

                # old_env = deepcopy(env)
                next_state, reward, terminal, _ = env.step(actions[0])
                energy_surface.append(np.sum(state) - np.sum(next_state))

                diffs = compute_layer_diff(state, next_state, stack_depth)
                # print(f"{diffs.sum()=}")
                if np.all(actions[0] == previous_action):
                    chosen_previous_action += 1

                n_annihilated_syndromes = np.absolute(
                    np.where(diffs > 0, diffs, 0).sum()
                )
                n_created_syndromes = np.absolute(np.where(diffs < 0, -diffs, 0).sum())
                if actions[0][-1] != TERMINAL_ACTION:
                    if n_annihilated_syndromes == 0 and n_created_syndromes == 0:
                        assert not np.all(
                            state == next_state
                        ), f"{state=}, \n{next_state=}, \n{actions[0]=}"

                syndromes_annihilated[j_episode] += n_annihilated_syndromes
                syndromes_created[j_episode] += n_created_syndromes
                # TODO count number of corrected errors
                state = next_state
                previous_action = actions[0]
                mean_q_value_per_p_error = incremental_mean(
                    q_value, mean_q_value_per_p_error, steps_counter
                )

            # logger.info(
            #     f"{chosen_previous_action / (steps_counter - 1 + 1e-20)=}, {previous_action=}"
            # )
            if plot_one_episode and i_err_list == 0 and j_episode == 0:
                env.render(block=True)

            # theoretical_q_value = compute_theoretical_q_value(energy_toric)
            mean_steps_per_p_error = incremental_mean(
                num_steps_per_episode, mean_steps_per_p_error, j_episode + 1
            )
            # TODO: need better measure for reaching terminal state
            fully_corrected[j_episode] = terminal_state
            _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                env.actual_errors, env.actions, env.vertex_mask, env.plaquette_mask
            )
            ground_state[j_episode] = _ground_state
            remaining_syndromes[j_episode] = n_syndromes
            logical_errors[j_episode] = n_loops
            # TODO: keep track of failed syndromes for later analysis
            # if not terminal_state or not _ground_state:
            #     # init qubit state
            #     failed_syndromes.append(
            #         create_syndrome_output_stack(env.actual_errors, env.vertex_mask, env.plaquette_mask)
            #     )

        assert np.any(syndromes_annihilated > 0), syndromes_annihilated
        assert np.any(syndromes_created > 0), syndromes_created
        fully_corrected_list[i_err_list] = (np.sum(fully_corrected)) / num_of_episodes
        ground_state_list[i_err_list] = (
            1 - (num_of_episodes - np.sum(ground_state)) / num_of_episodes
        )
        average_number_of_steps_list[i_err_list] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i_err_list] = np.round(mean_q_value_per_p_error, 3)

        remaining_syndromes_list[i_err_list] = np.mean(remaining_syndromes)
        syndromes_annihilated_list[i_err_list] = (
            np.mean(syndromes_annihilated) / average_number_of_steps_list[i_err_list]
        )
        syndromes_created_list[i_err_list] = (
            np.mean(syndromes_created) / average_number_of_steps_list[i_err_list]
        )
        logical_errors_list[i_err_list] = np.mean(logical_errors)

    syndrome_creation_normalization = (
        syndromes_annihilated_list + syndromes_created_list
    )
    assert np.all(
        syndrome_creation_normalization > 0
    ), f"{syndrome_creation_normalization=}"
    syndromes_annihilated_list /= syndrome_creation_normalization
    syndromes_created_list /= syndrome_creation_normalization

    results_number_per_episode = {
        "fully_corrected_per_episode": fully_corrected_list,
        "ground_state_per_episode": ground_state_list,
        "logical_errors_per_episode": logical_errors_list,
    }

    results_number_per_step = {
        "syndromes_annihilated_per_step": syndromes_annihilated_list,
        "syndromes_created_per_step": syndromes_created_list,
    }

    results_mean_per_p_error = {
        "avg_number_of_steps": average_number_of_steps_list,
        "mean_q_value": mean_q_list,
        "remaining_syndromes_per_episode": remaining_syndromes_list,
    }

    return (
        results_number_per_episode,
        results_number_per_step,
        results_mean_per_p_error,
    )
