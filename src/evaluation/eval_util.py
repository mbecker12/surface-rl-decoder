"""
Utility functions to help evaluate the performance
of a model.
"""
from copy import deepcopy
import logging
from typing import List, Tuple
import numpy as np

# pylint: disable=not-callable
import torch
from distributed.environment_set import EnvironmentSet
from evaluation.eval_init_utils import OUT_OF_RANGE
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import (
    SOLVED_EPISODE_REWARD,
    STATE_MULTIPLIER,
    STATE_MULTIPLIER_INVERSE,
    SYNDROME_DIFF_REWARD,
    check_repeating_action,
    create_syndrome_output_stack,
)

# define keys for different groups in the result dictionary
RESULT_KEY_EPISODE = "per_episode"
RESULT_KEY_STEP = "per_step"
RESULT_KEY_P_ERR = "per_p_err"
RESULT_KEY_HISTOGRAM_Q_VALUES = "all_q_values"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


def create_user_eval_state(
    env: SurfaceCode,
    idx_episode: int,
    discount_factor_gamma=0.9,
    discount_intermediate_reward=0.3,
    annealing_intermediate_reward=1.0,
    punish_repeating_actions: int = 0,
) -> Tuple[np.ndarray, List, float]:
    """
    Create a state from a predefined qubit error configuration
    to have some episodes that we can analyze and compare with the
    expected optimal outcome.

    Parameters
    ==========
    env: instance of the surface code class
    idx_episode: index to determine which of the user-defined qubit
        configurations to choose; starting at 0
    discount_factor_gamma: gamma factor in reinforcement learning,
        to discount the effect of steps in the future
    discount_intermediate_reward: (optional) discount factor determining how much
        early layers should be discounted when calculating the intermediate reward
    annealing_intermediate_reward: (optional) variable that should decrease over time during
        a training run to decrease the effect of the intermediate reward
    punish_repeating_actions: (optional) (1 or 0) flag acting as multiplier to
        enable punishment for repeating actions that already exist in the action history

    Returns
    =======
    state: the syndrome stack for the manaully created qubit configuration
    expected_actions: list of optimal actions for the given qubit configuration
    theoretical_max_q_value: manually calculated q value
        if the optimal action is chosen. Is only valid for the very first step
        taken
    """
    env.reset()
    stack_depth = env.stack_depth
    code_size = env.code_size
    (
        env.qubits,
        expected_actions,
        theoretical_max_q_value,
    ) = provide_deterministic_qubit_errors(
        idx_episode,
        stack_depth,
        code_size,
        discount_factor_gamma=discount_factor_gamma,
        annealing_intermediate_reward=annealing_intermediate_reward,
    )
    env.actual_errors = deepcopy(env.qubits)
    env.state = (
        create_syndrome_output_stack(env.qubits, env.vertex_mask, env.plaquette_mask)
        * STATE_MULTIPLIER
    )
    env.syndrome_errors = np.zeros_like(env.state, dtype=bool)

    return env.state, expected_actions, theoretical_max_q_value


def provide_deterministic_qubit_errors(
    index,
    stack_depth,
    code_size,
    discount_factor_gamma=0.9,
    annealing_intermediate_reward=1.0,
):
    """
    Provide a selection of manually-created qubit error configurations
    for use in a deterministic evaluation routine.

    Parameters
    ==========
    index: determines which qubit configuration to choose, starting at 0
    stack_depth: height of the syndrome stack, h
    code_size: number of rows/columns in the syndrome stack, usually d+1
    discount_factor_gamma: gamma factor in reinforcement learning,
        to discount the effect of steps in the future
    annealing_intermediate_reward: (optional) variable that should decrease over time during
        a training run to decrease the effect of the intermediate reward

    Return
    ======
    qubits: qubit stack
    expected_actions: list of optimal actions for the given qubit configuration
    theoretical_max_q_value: manually calculated q value
        if the optimal action is chosen. Is only valid for the very first step
        taken
    """
    qubits = np.zeros((stack_depth, code_size, code_size), dtype=np.uint8)

    # single X error
    if index == 0:
        halfway_point = int(code_size / 2)
        qubits[-1, halfway_point, halfway_point] = 1
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 2 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(halfway_point, halfway_point, 1)]

    if index == 1:
        qubits[-1, 0, 0] = 1
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(0, 0, 1)]

    # single Z error
    if index == 2:
        halfway_point = int(code_size / 2)
        qubits[-1, halfway_point, halfway_point] = 3
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 2 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(halfway_point, halfway_point, 3)]

    if index == 3:
        qubits[-1, 0, 0] = 3
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(0, 0, 3)]

    # one X and one Z error
    if index == 4:
        qubits[-1, 0, code_size - 1] = 1
        qubits[-1, code_size - 1, 0] = 3
        theoretical_max_q_value = (
            1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
            + discount_factor_gamma
            * (
                SOLVED_EPISODE_REWARD * discount_factor_gamma
                + 1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
            )
        )
        expected_actions = [(0, code_size - 1, 1), (code_size - 1, 0, 3)]

    if index == 5:
        qubits[-1, 0, code_size - 1] = 3
        qubits[-1, code_size - 1, 0] = 1
        expected_actions = [(0, code_size - 1, 3), (code_size - 1, 0, 1)]
        theoretical_max_q_value = (
            1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
            + discount_factor_gamma
            * (
                SOLVED_EPISODE_REWARD * discount_factor_gamma
                + 1 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
            )
        )
    # single Y error
    if index == 6:
        halfway_point = int(code_size / 2)
        qubits[-1, halfway_point, halfway_point] = 2
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 4 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(halfway_point, halfway_point, 2)]

    if index == 7:
        qubits[-1, 0, 0] = 2
        theoretical_max_q_value = (
            SOLVED_EPISODE_REWARD * discount_factor_gamma
            + 2 * annealing_intermediate_reward * SYNDROME_DIFF_REWARD
        )
        expected_actions = [(0, 0, 2)]

    return qubits, expected_actions, theoretical_max_q_value


def calculate_theoretical_max_q_value(state, gamma, discount_inter_reward):
    """
    Approximately calculate the optimal q value for a given state.
    The best possible q value should be
    when annihilating at least one syndrome
    with one action until no syndromes remain.
    Takes into account the annihilation of syndromes depthwise
    with one action.
    For simplicity, this has to disregard
    syndrome measurement errors which are
    present in the state.
    Also, this further approximates the real possible q value by
    only looking at average syndrome depth.

    Parameters
    ==========
    state: the syndrome stack for the manaully created qubit configuration
    gamma: gamma factor in reinforcement learning,
        to discount the effect of steps in the future
    discount_inter_reward: discount factor, to discount the depth of a syndrome,
        has the effect that, the closer to the top layer an annihiliation of syndromes
        takes place, the more it contributes

    Returns
    =======
    q_value: q value if the optimal action is chosen
    """
    n_syndromes_last_layer = np.sum(state[-1]) * STATE_MULTIPLIER_INVERSE
    # assume that syndromes (in the final layer) always come in pairs
    # this disregards edge qubits, where one qubit will only cause
    # one syndrome
    # Besides, this ignores the fact that Y errors cause more syndromes
    n_required_actions = np.ceil(n_syndromes_last_layer / 2)
    n_syndromes_total = np.sum(state) * STATE_MULTIPLIER_INVERSE
    stack_depth = state.shape[-3]

    avg_syndrome_depth = (
        int(n_syndromes_total / n_syndromes_last_layer)
        if n_syndromes_last_layer > 0
        else 0
    )

    one_minus_inter_rew_discount_inv = 1.0 / (1.0 - discount_inter_reward)
    one_minus_gamma_inv = 1.0 / (1.0 - gamma)

    # avg reward per correct action for annihilating syndromes
    inter_reward = (
        2
        * SYNDROME_DIFF_REWARD
        * STATE_MULTIPLIER
        * (1.0 - discount_inter_reward ** avg_syndrome_depth)
        * one_minus_inter_rew_discount_inv
    )

    # avg punishment for creating syndromes at the bottom layers
    inter_punishment = (
        2
        * SYNDROME_DIFF_REWARD
        * STATE_MULTIPLIER
        * (
            discount_inter_reward ** avg_syndrome_depth
            - discount_inter_reward ** stack_depth
        )
        * one_minus_inter_rew_discount_inv
    )

    inter_reward -= inter_punishment

    # add all the required actions together
    final_reward = (
        inter_reward * (1.0 - gamma ** n_required_actions) * one_minus_gamma_inv
    )
    # add the final reward for successfully terminating a solved episode
    final_reward += SOLVED_EPISODE_REWARD * gamma ** n_required_actions
    return final_reward


def average_episode_metric(
    value_array, n_environments, finished_array, extra_values_array: List = [0]
):
    """
    Calculate the average of an episode-based metric (e.g. number of remaining
    syndromes per episode) for evaluation routines.
    This function takes into account if an environment
    has already finished one or more times within a fixed
    number of steps and adjusts the average value accordingly.

    Parameters
    ==========
    value_array: array containing the accumulated values for each episode,
        shape: (n_environments, )
    n_environments: number of (possibly parallel) environments, independent
        of the number of episodes across all environments
    finished_array: array that denotes for each environment
        how often that environment has finished
    extra_values_array: (optional) some values that are to be saved
        in a separate list because they cannot be safely aggregated to
        the value_array

    Returns
    =======
    avg_value: The average of the accumulated metric
    """
    # NOTE: aware of linter hint about dangerous default value.
    # Actually, we're not changing the argument, only using the sum of it,
    # so it won't alter the content passed as that argument
    if np.any(finished_array):
        denominator_inv = 1.0 / (n_environments + np.sum(finished_array))
        avg_value = (np.sum(value_array) + np.sum(extra_values_array)) * denominator_inv
    else:
        avg_value = np.mean(value_array)

    return avg_value


def count_spikes(arr, verbosity=0):
    """
    Count spikes in an array. Each spike should hint at unnecessary actions
    that undo the effects of previous actions.

    With this, one could example look at whether the intermediate reward
    goes back and forth during an episode if only one action was chosen
    for the whole episode.

    Parameters
    ==========
    arr: list-like container with evaluation metrics to analyze

    Returns
    =======
    spikiness: measure between 0 (monotonic) and 1 (reocurring up and down)
    """
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


def prepare_step(global_steps, terminals, steps_per_episode, states, device):
    """
    Utility function to prepare central quantites before executing a step in the
    environment.

    Gets information about current active environments, increases each active environment's
    step count, and converts the state to a torch tensor with which the neural
    network can work.
    """
    global_steps += 1

    is_active = np.argwhere(1 - terminals).flatten()
    steps_per_episode[is_active] += 1

    assert len(states.shape) == 4  # need batch, stack_depth, d+1, d+1
    torch_states = torch.tensor(states[is_active], dtype=torch.float32).to(device)

    return global_steps, torch_states, is_active, steps_per_episode


def reset_local_actions_and_qvalues(terminal_actions, empty_q_values):
    """
    Fill up all actions with terminal actions as the predetermined default
    and set up the q values with all zero q values.

    These values should be overwritten in all active environments with the
    correct actions and q values.
    """
    actions = terminal_actions
    q_values = empty_q_values

    return actions, q_values


def get_two_highest_q_values(q_values):
    """
    To get statistics about the q values and the q value certainty,
    we need to extract the two highest q values proposed by the model.
    """
    # gives the indices of the highest values for each row
    top_idx = np.argpartition(q_values, (-2, -1), axis=1)[:, -2:]
    highest_q_values = q_values[np.arange(q_values.shape[0])[:, None], top_idx]
    q_value = highest_q_values[:, -1]
    second_q_value = highest_q_values[:, -2]

    return q_value, second_q_value


def aggregate_q_value_stats(
    q_value_aggregation,
    q_value_diff_aggregation,
    q_value_certainty_aggregation,
    terminal_q_value_aggregation,
    q_value,
    second_q_value,
    theoretical_q_values,
    terminal_q_value,
):
    """
    Add q value statistics to the aggregation variables.

    Call this function in each step and for each active environment.
    """
    q_value_aggregation += q_value
    q_value_diff_aggregation += q_value - theoretical_q_values
    q_value_certainty_aggregation += q_value - second_q_value
    terminal_q_value_aggregation += terminal_q_value

    return (
        q_value_aggregation,
        q_value_diff_aggregation,
        q_value_certainty_aggregation,
        terminal_q_value_aggregation,
    )


def calc_theoretical_q_value(
    is_user_episode,
    steps_per_episode,
    theoretical_q_values,
    states,
    discount_factor_gamma,
    discount_intermediate_reward,
):
    """
    Calculate an approximated theoretical q value.

    First, determine if a recalculation is necessary based on whether
    the episode is active and a random episode or a user-generated episode
    beyond the first step.

    The calculation is based on an approximated average syndrome depth
    and assumes two syndromes for every qubit error.
    """
    recalculate_theoretical_value_mask = np.logical_not(
        np.logical_and(is_user_episode, steps_per_episode == 1)
    )

    theoretical_q_values[recalculate_theoretical_value_mask] = np.array(
        [
            calculate_theoretical_max_q_value(
                states[i], discount_factor_gamma, discount_intermediate_reward
            )
            for i in np.where(recalculate_theoretical_value_mask)[0]
        ]
    )

    return theoretical_q_values


# pylint: disable=too-many-arguments
def prepare_user_episodes(
    states,
    expected_actions_per_episode,
    theoretical_q_values,
    total_n_episodes,
    num_of_random_episodes,
    num_of_user_episodes,
    env_set: EnvironmentSet,
    discount_factor_gamma=0.9,
    discount_intermediate_reward=0.3,
    annealing_intermediate_reward=1,
    punish_repeating_actions=0,
):
    """
    Load predetermined user episodes
    and prepare all surrounding quantities that come with it.
    """
    # prepare masks to filter user_episodes
    is_user_episode = np.zeros(total_n_episodes, dtype=int)
    is_user_episode[num_of_random_episodes:] = 1

    # prepare user defined episodes
    for j_user_episode in range(num_of_user_episodes):
        j_all_episodes = j_user_episode + num_of_random_episodes
        state, expected_actions, theoretical_q_value = create_user_eval_state(
            env_set.environments[j_all_episodes],
            j_user_episode,
            discount_factor_gamma=discount_factor_gamma,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )
        states[j_all_episodes] = state
        expected_actions_per_episode[j_user_episode] = expected_actions
        theoretical_q_values[j_all_episodes] = theoretical_q_value

    return states, expected_actions_per_episode, theoretical_q_values, is_user_episode


def check_correct_actions(
    actions,
    expected_actions_per_episode,
    correct_actions_aggregation,
    total_n_episodes,
    num_of_random_episodes,
    num_of_user_episodes,
):
    """
    Check whether the proposed actions match the expected, most optimal
    actions defined for user-defined episodes.
    """
    # check user-defined / expected actions
    # in user episodes
    correct_actions_all = np.array(
        [
            check_repeating_action(
                actions[i],
                expected_actions_per_episode[i - num_of_random_episodes],
                len(expected_actions_per_episode[i - num_of_random_episodes]),
            )
            for i in range(total_n_episodes - num_of_user_episodes, total_n_episodes)
        ]
    )
    correct_actions_aggregation += correct_actions_all

    return correct_actions_aggregation


def count_array_raises(arr):
    """
    Get a metric measuring how often an array made up of a sequence of numbers
    shows a positive slope.
    """
    num_raises = len(np.argwhere(np.diff(arr) > 0))
    return 2 * num_raises / len(arr)


def get_energy_stats(energies):
    """
    Get different statistics for the accumulated energy statistics from
    every step.
    """
    energies = energies[energies > OUT_OF_RANGE]
    energy_spikes = count_spikes(energies)
    energy_raises = count_array_raises(energies)
    energy_final = energies[-1]
    energy_difference = energy_final - energies[0]

    return energy_spikes, energy_raises, energy_final, energy_difference


def get_intermediate_reward_stats(inter_rewards):
    """
    Get different statistics for the accumulated statistics about
    intermediate rewards from every step.
    """
    inter_rewards = inter_rewards[inter_rewards > OUT_OF_RANGE]
    inter_rew_spikes = count_spikes(inter_rewards)
    negatives = np.where(inter_rewards < 0, inter_rewards, -99999).flatten()
    num_negative_inter_rewards = len(np.argwhere(-5000 < negatives).flatten())
    # num_negative_inter_rewards = len(np.argwhere(-5000 < inter_rewards < 0))

    positive_rewards = inter_rewards > 0
    if len(np.argwhere(positive_rewards)) > 0:
        avg_positive_inter_rew = np.mean(inter_rewards[inter_rewards > 0])
    else:
        avg_positive_inter_rew = 0

    min_inter_rew = np.min(inter_rewards)

    return (
        inter_rew_spikes,
        num_negative_inter_rewards,
        avg_positive_inter_rew,
        min_inter_rew,
    )
