"""
Define an evaluation routine to keep track of the agent's ability
to decode syndromes
"""
import logging
import numpy as np
import torch
from distributed.util import (
    action_to_q_value_index,
    assert_not_all_elements_equal,
    incremental_mean,
    select_actions,
)
from distributed.learner_util import (
    calculate_theoretical_max_q_value,
    create_user_eval_state,
)
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import (
    TERMINAL_ACTION,
    check_final_state,
    check_repeating_action,
    compute_layer_diff,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-locals, too-many-statements
def evaluate(
    model,
    env,
    device,
    p_error_list,
    p_msmt_list,
    num_of_episodes=8,
    num_of_user_episodes=8,
    epsilon=0.0,
    num_of_steps=50,
    plot_one_episode=True,
    discount_factor_gamma=0.9,
):
    """
    Evaluate the current policy.
    """
    # pylint: disable=not-callable
    # for torch.tensor()
    model.eval()

    total_num_of_episodes = num_of_episodes + num_of_user_episodes

    env = SurfaceCode()
    system_size = env.system_size
    stack_depth = env.stack_depth
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
    mean_q_diff_list = np.zeros(len(p_error_list))
    chose_correct_action_list = np.zeros(len(p_error_list)) - 1

    annealing_intermediate_reward = 1.0
    discount_intermediate_reward = 0.75
    punish_repeating_actions = 0

    for i_err_list, p_error in enumerate(p_error_list):
        # initialize arrays to keep track of scoring metrics for each episode
        p_msmt = p_msmt_list[i_err_list]
        ground_state = np.ones(total_num_of_episodes, dtype=bool)
        fully_corrected = np.zeros(total_num_of_episodes, dtype=bool)
        syndromes_annihilated = np.zeros(total_num_of_episodes, dtype=int)
        syndromes_created = np.zeros(total_num_of_episodes, dtype=int)
        remaining_syndromes = np.zeros(total_num_of_episodes, dtype=int)
        logical_errors = np.zeros(total_num_of_episodes, dtype=int)
        chose_correct_actions = np.zeros(num_of_user_episodes, dtype=int)
        # initialize some average values, transcending episodes
        mean_steps_per_p_error = 0
        mean_q_value_per_p_error = 0
        steps_counter = 0
        mean_q_value_diff = 0
        common_actions = np.zeros(total_num_of_episodes)

        for j_episode in range(total_num_of_episodes):
            # initialize an episode
            steps_counter = 0
            logger.debug(f"{p_error=}, episode: {j_episode}")
            num_steps_per_episode = 0
            previous_action = (0, 0, 0)
            chosen_previous_action = 0
            terminal_state = 0
            terminal = False
            is_user_episode = False
            actions_in_one_episode = np.zeros(num_of_steps) - 1

            state = env.reset(p_error=p_error, p_msmt=p_msmt)

            # if desired, initialize some manually prepared episodes
            # to get a better look at how the agent fares
            # in an environment where we know the optimal solution
            # deterministically
            if j_episode >= num_of_episodes:
                is_user_episode = True
                correct_actions = 0
                state, expected_actions, theoretical_q_value = create_user_eval_state(
                    env,
                    j_episode - num_of_episodes,
                    discount_factor_gamma=discount_factor_gamma,
                    discount_intermediate_reward=discount_intermediate_reward,
                    annealing_intermediate_reward=annealing_intermediate_reward,
                    punish_repeating_actions=punish_repeating_actions,
                )

            torch_state = torch.tensor(state, dtype=torch.float32).to(device)
            torch_state = torch.unsqueeze(torch_state, 0)

            while not terminal and num_steps_per_episode < num_of_steps:
                # iterate through the episode
                steps_counter += 1
                num_steps_per_episode += 1

                # let the agent do its work
                actions, q_values = select_actions(
                    torch_state, model, system_size, epsilon=epsilon
                )
                q_value_idx = action_to_q_value_index(actions[0], system_size)
                actions_in_one_episode[num_steps_per_episode - 1] = q_value_idx

                # check if we're in a user-prepared episode
                # and the agent's action agrees with our expectations
                if is_user_episode and check_repeating_action(
                    actions[0], expected_actions, len(expected_actions)
                ):
                    correct_actions += 1

                # calculate q values and difference
                # between theoretical and empirical q values.
                # keep in mind that theoretical q values are just approximations,
                q_value_index = action_to_q_value_index(actions[0], system_size)
                q_value = q_values[0, q_value_index]

                # override theoretical q_value if it's not a user episode.
                # if it's a user episode, override the theor. q value after the first step
                if not (is_user_episode and num_steps_per_episode == 1):
                    theoretical_q_value = calculate_theoretical_max_q_value(
                        state, discount_factor_gamma
                    )

                q_value_diff = q_value - theoretical_q_value
                mean_q_value_diff = incremental_mean(
                    q_value_diff, mean_q_value_diff, steps_counter
                )

                # apply the chosen action
                next_state, _, terminal, _ = env.step(
                    actions[0],
                    discount_intermediate_reward=discount_intermediate_reward,
                    annealing_intermediate_reward=annealing_intermediate_reward,
                    punish_repeating_actions=punish_repeating_actions,
                )

                # analyze how many syndromes were created or destroyed
                diffs = compute_layer_diff(state, next_state, stack_depth)
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

                # prepare next iteration in episode
                state = next_state
                previous_action = actions[0]
                mean_q_value_per_p_error = incremental_mean(
                    q_value, mean_q_value_per_p_error, steps_counter
                )
            # end while; iterate through one episode
            if plot_one_episode and i_err_list == 0 and j_episode == 0:
                env.render(block=True)

            # take the most-occuring action/q_value_idx in common_actions
            unique, counts = np.unique(actions_in_one_episode, return_counts=True)
            most_common_qvalue_idx = np.argmax(counts)
            most_common_qvalue = unique[most_common_qvalue_idx]
            common_actions[j_episode] = most_common_qvalue

            # update and/or save metrics
            mean_steps_per_p_error = incremental_mean(
                num_steps_per_episode, mean_steps_per_p_error, j_episode + 1
            )

            fully_corrected[j_episode] = terminal_state
            _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                env.actual_errors, env.actions, env.vertex_mask, env.plaquette_mask
            )
            ground_state[j_episode] = _ground_state
            remaining_syndromes[j_episode] = n_syndromes
            logical_errors[j_episode] = n_loops

            if is_user_episode:
                correct_actions /= steps_counter
                chose_correct_actions[j_episode - num_of_episodes] = correct_actions
        # end for; episodes

        # gather information about all episodes
        assert np.any(syndromes_annihilated > 0), syndromes_annihilated
        assert np.any(syndromes_created > 0), syndromes_created
        fully_corrected_list[i_err_list] = (
            np.sum(fully_corrected) / total_num_of_episodes
        )
        ground_state_list[i_err_list] = np.sum(ground_state) / total_num_of_episodes
        average_number_of_steps_list[i_err_list] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i_err_list] = np.round(mean_q_value_per_p_error, 3)
        mean_q_diff_list[i_err_list] = np.round(mean_q_value_diff, 3)

        unique = np.unique(common_actions)
        if not len(unique) > 1:
            logger.warning("Warning! Only one action was chosen in all episodes.")

        remaining_syndromes_list[i_err_list] = np.mean(remaining_syndromes)
        syndromes_annihilated_list[i_err_list] = (
            np.mean(syndromes_annihilated) / average_number_of_steps_list[i_err_list]
        )
        syndromes_created_list[i_err_list] = (
            np.mean(syndromes_created) / average_number_of_steps_list[i_err_list]
        )
        logical_errors_list[i_err_list] = np.mean(logical_errors)
        if num_of_user_episodes > 0:
            chose_correct_action_list[i_err_list] = np.mean(chose_correct_actions)
    # end for; error_list

    # calculate how many syndromes are
    # annihilated/created per step
    syndrome_creation_normalization = (
        syndromes_annihilated_list + syndromes_created_list
    )
    assert np.all(
        syndrome_creation_normalization > 0
    ), f"{syndrome_creation_normalization=}"
    syndromes_annihilated_list /= syndrome_creation_normalization
    syndromes_created_list /= syndrome_creation_normalization

    # group the evaluation metrics for output
    results_number_per_episode = {
        "fully_corrected_per_episode": fully_corrected_list,
        "ground_state_per_episode": ground_state_list,
        "logical_errors_per_episode": logical_errors_list,
    }

    results_number_per_step = {
        "syndromes_annihilated_per_step": syndromes_annihilated_list,
        "syndromes_created_per_step": syndromes_created_list,
        "correct_actions_per_step": chose_correct_action_list,
    }

    results_mean_per_p_error = {
        "avg_number_of_steps": average_number_of_steps_list,
        "mean_q_value": mean_q_list,
        "mean_q_value_difference": mean_q_diff_list,
        "remaining_syndromes_per_episode": remaining_syndromes_list,
    }

    return (
        results_number_per_episode,
        results_number_per_step,
        results_mean_per_p_error,
    )
