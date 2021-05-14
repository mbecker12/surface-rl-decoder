"""
Provide a script like function to perform evaluation
on multiple episodes in parallel.
"""
import os
import logging
from copy import deepcopy
import traceback
import numpy as np
import torch
from distributed.util import (
    action_to_q_value_index,
    q_value_index_to_action,
    select_actions,
)
from distributed.environment_set import EnvironmentSet
from evaluation.eval_util import (
    aggregate_q_value_stats,
    calc_theoretical_q_value,
    check_correct_actions,
    get_energy_stats,
    get_intermediate_reward_stats,
    get_two_highest_q_values,
    initialize_states_for_eval,
    prepare_step,
    prepare_user_episodes,
    reset_local_actions_and_qvalues,
)
from evaluation.eval_init_utils import (
    initialize_accumulation_stats,
    initialize_avg_containers,
    initialize_empty_containers,
)
from surface_rl_decoder.surface_code_util import (
    STATE_MULTIPLIER_INVERSE,
    TERMINAL_ACTION,
    check_final_state,
    compute_layer_diff,
)
from surface_rl_decoder.surface_code import SurfaceCode

# define keys for different groups in the result dictionary
RESULT_KEY_EPISODE_AVG = "avg_per_episode"
RESULT_KEY_EPISODE_MEDIAN = "median_per_episode"
RESULT_KEY_EPISODE = "per_episode"
RESULT_KEY_STEP = "per_step"
RESULT_KEY_Q_VALUE_STATS = "q_value_stats"
RESULT_KEY_HISTOGRAM_Q_VALUES = "all_q_values"
RESULT_KEY_AVG_INCREASING = "avg_increasing_counts"
RESULT_KEY_MEDIAN_INCREASING = "median_increasing_counts"
RESULT_KEY_INCREASING = "increasing_counts"
RESULT_KEY_ENERGY = "counts_energy"

RESULT_KEY_AVG_COUNTS = "avg_counts"
RESULT_KEY_MEDIAN_COUNTS = "median_counts"
RESULT_KEY_COUNTS = "counts"
RESULT_KEY_AVG_RATES = "avg_rates"
RESULT_KEY_MEDIAN_RATES = "median_rates"
RESULT_KEY_RATES = "rates"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-statements, too-many-branches, too-many-locals, too-many-arguments
def batch_evaluation(
    model,
    environment_def,
    device,
    num_of_random_episodes=8,
    num_of_user_episodes=8,
    epsilon=0.0,
    max_num_of_steps=50,
    discount_factor_gamma=0.9,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.3,
    punish_repeating_actions=0,
    p_err=0.0,
    p_msmt=0.0,
    verbosity=0,
    rl_type="q",
    post_run=False,
    code_size=None,
    stack_depth=None,
):
    """
    Run evaluation on multiple episodes.
    Gather metrics about the proficiency of the agent being evaluated.

    Returns
    =======
    (metrics dict, q values dict)
    metrics dict:
        Dictionary containing different groups of metrics
    q values dict:
        Dictionary containing a histogram describing the occurence of different q values
    """

    model.eval()
    # initialize environments for different episodes
    total_n_episodes = num_of_random_episodes + num_of_user_episodes

    if code_size is not None:
        os.environ["CONFIG_ENV_SIZE"] = str(code_size)
    if stack_depth is not None:
        os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)

    if environment_def is None or environment_def == "":
        environment_def = SurfaceCode(code_size=code_size, stack_depth=stack_depth)

    env_set = EnvironmentSet(environment_def, total_n_episodes)
    code_size = env_set.code_size
    stack_depth = env_set.stack_depth

    if post_run:
        print("Run Post-run analysis")
        num_errors = int(np.ceil(p_err * code_size * code_size))
        all_qubits, all_states = initialize_states_for_eval(
            n_environments=total_n_episodes,
            code_size=code_size,
            stack_depth=stack_depth,
            num_errors=num_errors
        )
        states = env_set.post_run_eval_reset_all(all_qubits, all_states)

        # print(env_set.environments[0].state)
        # print(env_set.environments[0].qubits)
        # print(env_set.environments[0].actions[:10])
        # print(env_set.environments[0].actual_errors)
    else:
        states = env_set.reset_all(
            np.repeat(p_err, total_n_episodes), np.repeat(p_msmt, total_n_episodes)
        )
        # print(env_set.environments[0].state)
        # print(env_set.environments[0].qubits)
        # print(env_set.environments[0].actions[:10])
        # print(env_set.environments[0].actual_errors)

    # # counts up to max_num_of_steps
    global_episode_steps = 0

    ## initialize stats
    # {
    #     terminals, ground_state, remaining_syndromes, logical_errors, q_value_aggregation,
    #     q_value_diff_aggregation, q_value_certainty_aggregation, correct_actions_aggregation,
    #     energy_spikes, energy_raises, energy_final, energy_difference,
    #     inter_rew_spikes, num_negative_inter_rew, mean_positive_inter_rew, min_inter_rew
    # }
    averages = initialize_avg_containers(total_n_episodes)

    # {
    #     syndromes_annihilated, syndromes_created, common_actions, correct_actions_aggregation,
    #     energies, intermediate_rewards, terminal_energies
    # }
    accumulators = initialize_accumulation_stats(
        total_n_episodes, num_of_user_episodes, max_num_of_steps
    )

    # {
    #     steps_per_episode, terminals, empty_actions, terminal_actions,
    #     empty_q_values, theoretical_q_values, expected_actions_per_episode,
    # }
    essentials = initialize_empty_containers(
        total_n_episodes, num_of_user_episodes, code_size
    )

    # take care of deterministic, user-defined episodes
    (
        states,
        essentials["expected_actions_per_episode"],
        essentials["theoretical_q_values"],
        is_user_episode,
    ) = prepare_user_episodes(
        states,
        essentials["expected_actions_per_episode"],
        essentials["theoretical_q_values"],
        total_n_episodes,
        num_of_random_episodes,
        num_of_user_episodes,
        env_set,
    )

    actions_in_one_episode = np.zeros((total_n_episodes, max_num_of_steps)) - 1

    if verbosity >= 4:
        all_q_values = np.zeros(
            (max_num_of_steps, total_n_episodes, 3 * code_size ** 2 + 1)
        )
    else:
        all_q_values = None

    terminals = essentials["terminals"]
    # iterate through episodes
    while global_episode_steps < max_num_of_steps:
        (
            global_episode_steps,
            torch_states,
            is_active,
            essentials["steps_per_episode"],
        ) = prepare_step(
            global_episode_steps,
            terminals,
            essentials["steps_per_episode"],
            states,
            device,
        )

        assert len(is_active.shape) <= 2, is_active.shape

        energies = (
            np.sum(np.sum(states[:, -1, :, :], axis=2), axis=1)
            * STATE_MULTIPLIER_INVERSE
        )
        assert energies.shape == (total_n_episodes,), energies.shape
        accumulators["energies"][:, global_episode_steps - 1] = energies

        actions, q_values, values, entropies = reset_local_actions_and_qvalues(
            essentials["terminal_actions"], essentials["empty_q_values"], essentials["empty_values"], essentials["empty_entropies"], rl_type=rl_type
        )

        # evaluate active episodes
        if "q" in rl_type.lower():
            tmp_actions, tmp_q_values = select_actions(
                torch_states, model, code_size, epsilon=epsilon
            )
        elif "ppo" in rl_type.lower():
            if epsilon == 1:
                tmp_actions, logits, tmp_values = model.select_greedy_action_ppo(
                    torch_states, return_logits=True, return_values=True
                )
            else:
                tmp_actions, logits, tmp_values = model.select_action_ppo(
                    torch_states, return_logits=True, return_values=True
                )

            # if len(tmp_actions) == 0:
            #     break
            tmp_actions = np.array(
                [q_value_index_to_action(action, code_size) for action in tmp_actions]
            )
            tmp_q_values = logits.detach().cpu().numpy()
            dist = torch.distributions.Categorical(logits=logits)
            tmp_entropies = dist.entropy().unsqueeze(1)
            try:
                values[is_active] = tmp_values.detach().cpu().numpy()
                entropies[is_active] = tmp_entropies.detach().cpu().numpy()
            except:
                error_traceback = traceback.format_exc()
                logger.error("Caught exception while trying to detach value and/or entropy array.")
                logger.error(error_traceback)
                logger.warning(f"{total_n_episodes=}, {is_active.shape=}, {tmp_values.shape=}, {tmp_entropies.shape=}, {logits.shape=}")
        else:
            logger.error(f"Error! Unknown RL type {rl_type}")
            raise Exception()

        actions[is_active] = tmp_actions
        q_values[is_active] = tmp_q_values
        
        if verbosity >= 4:
            all_q_values[global_episode_steps - 1, :, :] = q_values

        active_q_values = q_values[is_active]
        active_actions = actions[is_active]
        assert not np.all(actions[is_active][-1] == TERMINAL_ACTION), actions
        assert not np.all(active_actions[-1] == TERMINAL_ACTION), actions

        # revert action back to q value index fo later use
        q_value_indices = np.array(
            [action_to_q_value_index(actions[i], code_size) for i in is_active]
        )

        # TODO check correct actions
        # TODO check only active episodes
        accumulators["correct_actions_aggregation"] = check_correct_actions(
            actions,
            essentials["expected_actions_per_episode"],
            accumulators["correct_actions_aggregation"],
            total_n_episodes,
            num_of_random_episodes,
            num_of_user_episodes,
            is_active
        )

        first_q_value, second_q_value = get_two_highest_q_values(active_q_values)
        terminal_q_value = active_q_values[:, -1]

        theoretical_q_values = calc_theoretical_q_value(
            is_user_episode,
            essentials["steps_per_episode"],
            essentials["theoretical_q_values"],
            states,
            discount_factor_gamma,
            discount_intermediate_reward,
        )

        # get info about max q values
        actions_in_one_episode[is_active, global_episode_steps - 1] = q_value_indices
        if rl_type == "ppo":
            active_values = values[is_active]
            active_entropies = entropies[is_active]
        else:
            active_values = None
            active_entropies = None
        (
            accumulators["q_value_aggregation"][is_active],
            accumulators["q_value_diff_aggregation"][is_active],
            accumulators["q_value_certainty_aggregation"][is_active],
            accumulators["terminal_q_value_aggregation"][is_active],
            accumulators["values_aggregation"][is_active],
            accumulators["entropy_aggregation"][is_active]
        ) = aggregate_q_value_stats(
            accumulators["q_value_aggregation"][is_active],
            accumulators["q_value_diff_aggregation"][is_active],
            accumulators["q_value_certainty_aggregation"][is_active],
            accumulators["terminal_q_value_aggregation"][is_active],
            first_q_value,
            second_q_value,
            theoretical_q_values[is_active],
            terminal_q_value,
            values=active_values,
            entropies=active_entropies,
            values_aggregation=accumulators["values_aggregation"][is_active],
            entropy_aggregation=accumulators["entropy_aggregation"][is_active]
        )

        # apply the chosen action
        next_states, rewards, terminals, _ = env_set.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )

        # if an episode is not terminal, 'rewards' should only contain the intermediate rewards
        # we take care of the terminal episodes later
        accumulators["intermediate_rewards"][:, global_episode_steps - 1] = rewards

        # count the difference of number of syndromes in each layer
        diffs_all = np.array(
            [
                compute_layer_diff(states[i], next_states[i], stack_depth)
                for i in range(total_n_episodes)
            ]
        )
        # ... and infer how many syndromes were created or destroyed
        n_annihilated_syndromes = np.sum(diffs_all > 0, axis=1)
        n_created_syndromes = np.sum(diffs_all < 0, axis=1)
        assert n_annihilated_syndromes.shape == (
            total_n_episodes,
        ), n_annihilated_syndromes

        # if we have a non-terminal episode and no net syndrome
        # creation or annihilation, at least the state must have changed
        # by applying an action.
        # Otherwise something went really wrong here
        non_terminal_episodes = np.where(actions[:, -1] != TERMINAL_ACTION)[0]
        # print(f"{non_terminal_episodes=}")
        for j in non_terminal_episodes:
            if n_annihilated_syndromes[j] == 0 and n_created_syndromes[j] == 0:
                # make sure that if a qubit changing action was chosen,
                # the syndrome state changes from one step to the next
                assert not np.all(
                    states[j] == next_states[j]
                ), f"{states[j]=}, \n{next_states[j]=}, \n{actions[j]=}"

        accumulators["syndromes_annihilated"] += n_annihilated_syndromes[
            is_active
        ].sum()
        accumulators["syndromes_created"] += n_created_syndromes[is_active].sum()

        # print(f"t={global_episode_steps}, {is_active=}")
        # Handle terminal episodes:
        # terminal episodes will just be ignored
        # and wait for all episodes to finish or reach the max number of steps
        if np.any(terminals):
            indices = np.argwhere(terminals).flatten()
            # print(f"t={global_episode_steps}, terminal {indices=}")

            for i in indices:
                # skip episodes that are no longer active
                if i not in is_active:
                    # print(f"skip episode {i}")
                    continue

                accumulators["intermediate_rewards"][
                    indices, global_episode_steps - 1
                ] = np.repeat(0, len(indices))
                accumulators["terminal_energies"][indices] = energies[indices]

                _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                    env_set.environments[i].actual_errors,
                    env_set.environments[i].actions,
                    env_set.environments[i].vertex_mask,
                    env_set.environments[i].plaquette_mask,
                )
                # print(f"t={global_episode_steps}, {i=}, {_ground_state=}")
                averages["ground_state"][i] += _ground_state
                averages["remaining_syndromes"][i] += n_syndromes
                averages["logical_errors"][i] += n_loops

        if np.all(terminals):
            break

        states = next_states
        env_set.states = deepcopy(states)

    # end while; step through episode

    averages["averaging_terminals"] = terminals

    for i in range(total_n_episodes):
        # TODO: find a way to count number of successful episodes
        # mix of logical errors, remaining syndromes
        # ...is excatly what ground state measures
        unique, counts = np.unique(actions_in_one_episode[i], return_counts=True)
        terminal_action_idx = np.argwhere(unique == -1)
        counts[terminal_action_idx] = 1
        most_common_qvalue_idx = np.argmax(counts)
        most_common_qvalue = unique[most_common_qvalue_idx]
        accumulators["common_actions"][i] = most_common_qvalue

        (
            averages["energy_spikes"][i],
            averages["energy_raises"][i],
            averages["energy_final"][i],
            averages["energy_difference"][i],
        ) = get_energy_stats(accumulators["energies"][i])
        (
            averages["inter_rew_spikes"][i],
            averages["num_negative_inter_rew"][i],
            averages["mean_positive_inter_rew"][i],
            averages["min_inter_rew"][i],
        ) = get_intermediate_reward_stats(accumulators["intermediate_rewards"][i])

        if i not in is_active:
            continue

        _, _ground_state, (n_syndromes, n_loops) = check_final_state(
            env_set.environments[i].actual_errors,
            env_set.environments[i].actions,
            env_set.environments[i].vertex_mask,
            env_set.environments[i].plaquette_mask,
        )

        # print(f"{i=}, {_ground_state=} for real")
        averages["ground_state"][i] += _ground_state
        averages["remaining_syndromes"][i] += n_syndromes
        averages["logical_errors"][i] += n_loops
    # end for; loop over all episodes one last time

    print("End of Evaluation")
    # print(env_set.environments[0].state)
    # print(env_set.environments[0].qubits)
    # print(env_set.environments[0].actions[:10])
    # print(env_set.environments[0].actual_errors)

    avg_number_of_steps = np.mean(essentials["steps_per_episode"])
    avg_chose_correct_action_per_episode = np.mean(
        accumulators["correct_actions_aggregation"] / num_of_user_episodes
    )
    # print(f"{averages['ground_state']=}")
    # print(f"{averages['averaging_terminals']=}")
    avg_ground_state = np.mean(averages["ground_state"])
    avg_remaining_syndromes = np.mean(averages["remaining_syndromes"])
    avg_logical_errors = np.mean(averages["logical_errors"])
    avg_terminals = np.mean(averages["averaging_terminals"])
    avg_energy_spikes = np.mean(averages["energy_spikes"])
    avg_energy_raises = np.mean(averages["energy_raises"])
    avg_energy_final = np.mean(averages["energy_final"])
    avg_energy_difference = np.mean(averages["energy_difference"])
    avg_num_neg_inter_rew = np.mean(averages["num_negative_inter_rew"])
    avg_mean_positive_inter_rew = np.mean(averages["mean_positive_inter_rew"])
    avg_min_inter_rew = np.mean(averages["min_inter_rew"])

    median_number_of_steps = np.median(essentials["steps_per_episode"])
    median_chose_correct_action_per_episode = np.median(
        accumulators["correct_actions_aggregation"] / num_of_user_episodes
    )
    median_ground_state = np.median(averages["ground_state"])
    median_remaining_syndromes = np.median(averages["remaining_syndromes"])
    median_logical_errors = np.median(averages["logical_errors"])
    median_terminals = np.median(averages["averaging_terminals"])
    median_energy_spikes = np.median(averages["energy_spikes"])
    median_energy_raises = np.median(averages["energy_raises"])
    median_energy_final = np.median(averages["energy_final"])
    median_energy_difference = np.median(averages["energy_difference"])
    median_num_neg_inter_rew = np.median(averages["num_negative_inter_rew"])
    median_mean_positive_inter_rew = np.median(averages["mean_positive_inter_rew"])
    median_min_inter_rew = np.median(averages["min_inter_rew"])

    # normalize stats about syndrome creation
    # we want to have the probability for the agent
    # to create or destroy a syndrome in each step
    syndromes_normalization = (
        accumulators["syndromes_annihilated"] + accumulators["syndromes_created"]
    )
    accumulators["syndromes_created"] /= syndromes_normalization
    accumulators["syndromes_annihilated"] /= syndromes_normalization

    # q_value_aggregation contains info of every step in every episode
    mean_q_value = np.mean(
        accumulators["q_value_aggregation"] / essentials["steps_per_episode"]
    )
    if rl_type == "q":
        mean_q_value_diff = np.mean(
            accumulators["q_value_diff_aggregation"] / essentials["steps_per_episode"]
        )
    elif rl_type == "ppo":
        mean_q_value_diff = np.zeros_like(mean_q_value)
        mean_values = np.mean(accumulators["values_aggregation"] / essentials["steps_per_episode"])
        mean_entropies = np.mean(accumulators["entropy_aggregation"] / essentials["steps_per_episode"])
    else:
        raise Exception(f"RL Type {rl_type} is not supported!")

    std_q_value = np.std(
        accumulators["q_value_aggregation"] / essentials["steps_per_episode"]
    )
    q_value_certainty = np.mean(
        accumulators["q_value_certainty_aggregation"] / essentials["steps_per_episode"]
    )
    avg_terminal_q_value = np.mean(
        accumulators["terminal_q_value_aggregation"] / essentials["steps_per_episode"]
    )
    median_terminal_q_value = np.median(
        accumulators["terminal_q_value_aggregation"] / essentials["steps_per_episode"]
    )

    # An untrained network seems to choose a certain action lots of times
    # within an episode.
    # Make sure that it's at least a different action for different
    # state initializations.
    unique_actions = np.unique(accumulators["common_actions"])

    if not len(unique_actions) > 1:
        logger.debug(
            f"{accumulators['common_actions']=}, {accumulators['common_actions'].shape=}"
        )
        logger.warning(
            "Warning! Only one action was chosen in all episodes. "
            f"Most common action index: {unique_actions}"
        )

    if verbosity >= 4:
        all_q_values = all_q_values.flatten()

    
    if rl_type == "q":
        q_value_result_dict = {
            "mean_q_value": mean_q_value,
            "mean_q_value_difference": mean_q_value_diff,
            "std_q_value": std_q_value,
            "q_value_certainty": q_value_certainty,
            "avg_terminal_q_val": avg_terminal_q_value,
            "median_terminal_q_val": median_terminal_q_value,
        }
    elif rl_type == "ppo":
        q_value_result_dict = {
            "mean_q_value": mean_q_value,
            "std_q_value": std_q_value,
            "q_value_certainty": q_value_certainty,
            "avg_terminal_q_val": avg_terminal_q_value,
            "median_terminal_q_val": median_terminal_q_value,
            "mean_values": mean_values,
            "mean_entropy": mean_entropies
        }

    return (
        {
            RESULT_KEY_RATES: {
                "energy_spikiness": avg_energy_spikes,
                "median energy_spikiness": median_energy_spikes,
                "syndromes_annihilated_per_step": accumulators["syndromes_annihilated"],
                "terminals_per_env": avg_terminals,
                "median terminals_per_env": median_terminals,
                "ground_state_per_env": avg_ground_state,
                "median ground_state_per_env": median_ground_state,
            },
            RESULT_KEY_COUNTS: {
                "energy_difference": avg_energy_difference,
                "median energy_difference": median_energy_difference,
                "energy_final": avg_energy_final,
                "median energy_final": median_energy_final,
                "energy_raises": avg_energy_raises,
                "median energy_raises": median_energy_raises,
            },
            RESULT_KEY_ENERGY: {
                "number_of_steps": avg_number_of_steps,
                "median number_of_steps": median_number_of_steps,
                # "min_inter_rew": avg_min_inter_rew,
                # "median min_inter_rew": median_min_inter_rew,
                # "num_neg_inter_rew": avg_num_neg_inter_rew,
                # "median num_neg_inter_rew": median_num_neg_inter_rew,
                # "mean_pos_inter_rew": avg_mean_positive_inter_rew,
                # "median mean_pos_inter_rew": median_mean_positive_inter_rew,
            },
            RESULT_KEY_EPISODE: {
                "correct_actions_per_episode": avg_chose_correct_action_per_episode,
                "median correct_actions_per_episode": median_chose_correct_action_per_episode,
                "logical_errors_per_episode": avg_logical_errors,
                "median logical_errors_per_episode": median_logical_errors,
                "remaining_syndromes_per_episode": avg_remaining_syndromes,
                "median remaining_syndromes_per_episode": median_remaining_syndromes,
            },
            RESULT_KEY_Q_VALUE_STATS: q_value_result_dict,
        },
        {RESULT_KEY_HISTOGRAM_Q_VALUES: all_q_values},
    )
