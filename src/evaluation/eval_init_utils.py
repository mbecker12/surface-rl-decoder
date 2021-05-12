"""
Utility functions to initialize different groups of containers
and aggregators used in the evaluation routine.
"""
import numpy as np
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION

OUT_OF_RANGE = -9999


def initialize_avg_containers(total_n_episodes):
    """
    Initialize all containers which we will use to obtain average and median
    values later.
    """
    # initialize containers for mean metrics
    averaging_terminals = np.zeros(total_n_episodes, dtype=bool)

    ground_state = np.zeros(total_n_episodes)
    remaining_syndromes = np.zeros(total_n_episodes)
    logical_errors = np.zeros(total_n_episodes)

    energy_spikes = np.zeros(total_n_episodes)
    energy_raises = np.zeros(total_n_episodes)
    energy_final = np.zeros(total_n_episodes)
    energy_difference = np.zeros(total_n_episodes)
    inter_rew_spikes = np.zeros(total_n_episodes)
    num_negative_inter_rew = np.zeros(total_n_episodes)
    mean_positive_inter_rew = np.zeros(total_n_episodes)
    min_inter_rew = np.zeros(total_n_episodes)

    return {
        "averaging_terminals": averaging_terminals,
        "ground_state": ground_state,
        "remaining_syndromes": remaining_syndromes,
        "logical_errors": logical_errors,
        "energy_spikes": energy_spikes,
        "energy_raises": energy_raises,
        "energy_final": energy_final,
        "energy_difference": energy_difference,
        "inter_rew_spikes": inter_rew_spikes,
        "num_negative_inter_rew": num_negative_inter_rew,
        "mean_positive_inter_rew": mean_positive_inter_rew,
        "min_inter_rew": min_inter_rew,
    }


def initialize_accumulation_stats(
    total_n_episodes, num_of_user_episodes, max_num_of_steps
):
    """
    Initialize containers which we will use to accumulate stats after each
    step.
    """
    # initialize values to be accumulated across al episodes and steps
    syndromes_annihilated = 0
    syndromes_created = 0
    common_actions = np.zeros(total_n_episodes)
    correct_actions_aggregation = np.zeros(num_of_user_episodes)
    energies = np.zeros((total_n_episodes, max_num_of_steps)) + OUT_OF_RANGE
    # TODO: not colleting stats about terminal energy atm
    terminal_energies = np.zeros(total_n_episodes) + OUT_OF_RANGE
    # TODO: check implementation with OUT_OF_RANGE token,
    # seems to cause problems in some calculations
    intermediate_rewards = np.zeros((total_n_episodes, max_num_of_steps)) + OUT_OF_RANGE
    q_value_aggregation = np.zeros(total_n_episodes)
    q_value_diff_aggregation = np.zeros(total_n_episodes)
    q_value_certainty_aggregation = np.zeros(total_n_episodes)
    terminal_q_value_aggregation = np.zeros(total_n_episodes)

    entropy_aggregation = np.zeros((total_n_episodes, 1))
    values_aggregation = np.zeros((total_n_episodes, 1))

    return {
        "syndromes_annihilated": syndromes_annihilated,
        "syndromes_created": syndromes_created,
        "correct_actions_aggregation": correct_actions_aggregation,
        "common_actions": common_actions,
        "energies": energies,
        "intermediate_rewards": intermediate_rewards,
        "terminal_energies": terminal_energies,
        "q_value_aggregation": q_value_aggregation,
        "q_value_diff_aggregation": q_value_diff_aggregation,
        "q_value_certainty_aggregation": q_value_certainty_aggregation,
        "terminal_q_value_aggregation": terminal_q_value_aggregation,
        "values_aggregation": values_aggregation,
        "entropy_aggregation": entropy_aggregation,
    }


def initialize_empty_containers(total_n_episodes, num_of_user_episodes, code_size):
    """
    Initialize base quantities for the general execution and running
    of the different environments.
    """
    steps_per_episode = np.zeros(total_n_episodes)
    terminals = np.zeros(total_n_episodes, dtype=bool)

    empty_actions = np.zeros((total_n_episodes, 3), dtype=np.uint8)
    terminal_actions = empty_actions
    terminal_actions[:, -1] = TERMINAL_ACTION
    empty_q_values = np.zeros(
        (total_n_episodes, 3 * code_size * code_size + 1), dtype=float
    )
    # ppo
    empty_values = np.zeros((total_n_episodes, 1), dtype=float)
    empty_entropies = np.zeros((total_n_episodes, 1), dtype=float)

    theoretical_q_values = np.zeros(total_n_episodes)
    expected_actions_per_episode = {i: None for i in range(num_of_user_episodes)}

    return {
        "steps_per_episode": steps_per_episode,
        "terminals": terminals,
        "empty_actions": empty_actions,
        "terminal_actions": terminal_actions,
        "empty_q_values": empty_q_values,
        "empty_values": empty_values,
        "empty_entropies": empty_entropies,
        "theoretical_q_values": theoretical_q_values,
        "expected_actions_per_episode": expected_actions_per_episode,
    }
