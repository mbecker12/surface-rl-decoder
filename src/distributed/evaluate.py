import numpy as np
import torch
from torch import from_numpy
import gym
import logging
from distributed.util import action_to_q_value_index, incremental_mean, select_action
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import check_final_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


def evaluate(
    model,
    env,
    device,
    p_error_list,
    p_msmt_list,
    num_of_episodes=1,
    num_actions=3,
    epsilon=0.0,
    num_of_steps=50,
    plot_one_episode=True,
):
    """
    Evaluate the current policy.
    """

    model.to(device)
    model.eval()

    # TODO: publish gym environment
    # TODO: make environment configurable
    # env = gym.make(env, config=conf)

    env = SurfaceCode()
    system_size = env.system_size
    assert (
        system_size % 2 == 1
    ), "System size (i.e. number of qubits) needs to be an odd number."

    # initialize arrays for scoring metrics
    ground_state_list = np.zeros(len(p_error_list))
    error_corrected_list = np.zeros(len(p_error_list))
    average_number_of_steps_list = np.zeros(len(p_error_list))
    mean_q_list = np.zeros(len(p_error_list))
    failed_syndromes = []

    for i, p_error in enumerate(p_error_list):
        p_msmt = p_msmt_list[i]
        ground_state = np.ones(num_of_episodes, dtype=bool)
        error_corrected = np.zeros(num_of_episodes, dtype=bool)
        mean_steps_per_p_error = 0
        mean_q_value_per_p_error = 0
        steps_counter = 0

        for j in range(num_of_episodes):
            logger.info(f"{p_error=}, episode: {j}")
            num_steps_per_episode = 0
            previous_action = 0
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

                action, q_values = select_action(
                    torch_state, model, system_size, epsilon=epsilon
                )

                q_value_index = action_to_q_value_index(action, system_size)
                q_value = q_values[0, q_value_index]
                experimental_q_values.append(q_value)

                next_state, reward, terminal, _ = env.step(action)
                energy_surface.append(np.sum(state) - np.sum(next_state))

                state = next_state
                mean_q_value_per_p_error = incremental_mean(
                    q_value, mean_q_value_per_p_error, steps_counter
                )

            if plot_one_episode and i == 0 and j == 0:
                env.render(block=True)

            # theoretical_q_value = compute_theoretical_q_value(energy_toric)
            mean_steps_per_p_error = incremental_mean(
                num_steps_per_episode, mean_steps_per_p_error, j + 1
            )
            # TODO: need better measure for reaching terminal state
            error_corrected[j] = terminal_state
            _, _ground_state, *_ = check_final_state(
                env.actual_errors, env.actions, env.vertex_mask, env.plaquette_mask
            )
            ground_state[j] = _ground_state

            # TODO: keep track of failed syndromes for later analysis
            # if not terminal_state or not _ground_state:
            #     # init qubit state
            #     failed_syndromes.append(
            #         create_syndrome_output_stack(env.actual_errors, env.vertex_mask, env.plaquette_mask)
            #     )

        error_corrected_list[i] = (
            num_of_episodes - np.sum(~error_corrected)
        ) / num_of_episodes
        ground_state_list[i] = (
            1 - (num_of_episodes - np.sum(ground_state)) / num_of_episodes
        )
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i] = np.round(mean_q_value_per_p_error, 3)

    return (
        error_corrected_list,
        ground_state_list,
        average_number_of_steps_list,
        mean_q_list,
        None,
    )
