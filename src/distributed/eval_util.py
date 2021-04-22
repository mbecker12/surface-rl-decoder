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
from distributed.util import action_to_q_value_index, select_actions
from distributed.eval_init_utils import OUT_OF_RANGE
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import (
    SOLVED_EPISODE_REWARD,
    STATE_MULTIPLIER,
    STATE_MULTIPLIER_INVERSE,
    SYNDROME_DIFF_REWARD,
    TERMINAL_ACTION,
    check_final_state,
    check_repeating_action,
    compute_layer_diff,
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

# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
def run_evaluation_in_batches(
    model,
    environment_def,
    device,
    num_of_random_episodes=8,
    num_of_user_episodes=8,
    epsilon=0.0,
    max_num_of_steps=50,
    plot_one_episode=True,
    discount_factor_gamma=0.9,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.3,
    punish_repeating_actions=0,
    p_err=0.0,
    p_msmt=0.0,
    verbosity=0,
):
    """
    Run some evaluation episodes for fixed p_error and p_msmt.
    Enables the states from different episodes to be processed
    by the neural network at the same time (as a batch).

    Parameters
    ==========
    model: (subclass of torch.nn.Module) Neural network model
    environment_def: (str or gym.Env) either environment name, or object
    device: torch.device
    num_of_random_episodes: number of episodes to with fully randomly generated states
    num_of_user_episodes: number of user-creates and predefined episodes,
        taken from create_user_eval_state(), hence this number is limited by the number of
        availabe examples in the helper function
    epsilon: probability of the agent choosing a random action
    max_num_of_steps: maximum number of steps per environment
    plot_one_episode: whether or not to render an example episode
    discount_factor_gamma: gamma / discount factor in reinforcement learning
    p_err: error rate of physical errors
    p_msmt: error rate of syndrome measurement errors
    annealing_intermediate_reward: (optional) variable that should decrease over time during
        a training run to decrease the effect of the intermediate reward
    punish_repeating_actions: (optional) (1 or 0) flag acting as multiplier to
        enable punishment for repeating actions that already exist in the action history
    discount_intermediate_reward: (optional) discount factor determining how much
        early layers should be discounted when calculating the intermediate reward
    verbosity: (int) verbosity level

    Returns
    =======
    Dictionary of three different metrics categories:

    per_episode:
    {
        n_terminals,
        n_ground_state,
        n_logical_errors,
        n_chose_correct_action_per_episode,
    },
    per_step:
    {
        syndromes_annihilated,
        syndromes_created
    },
    per_p_error:
    {
        avg_number_of_steps,
        mean_q_value,
        mean_q_value_diff,
        n_remaining_syndromes
    }
    """
    model.eval()
    # initialize environments for different episodes
    total_n_episodes = num_of_random_episodes + num_of_user_episodes
    if environment_def is None or environment_def == "":
        environment_def = SurfaceCode()

    # print(f"{environment_def.p_error=}")
    env_set = EnvironmentSet(environment_def, total_n_episodes)
    code_size = env_set.code_size
    syndrome_size = code_size + 1
    stack_depth = env_set.stack_depth
    states = env_set.reset_all(
        np.repeat(p_err, total_n_episodes), np.repeat(p_msmt, total_n_episodes)
    )

    # initialize containers for mean metrics
    steps_per_episode = np.zeros(total_n_episodes)
    expected_actions_per_episode = {i: None for i in range(num_of_user_episodes)}
    theoretical_q_values = np.zeros(total_n_episodes)
    common_actions = np.zeros(total_n_episodes)
    terminals = np.zeros(total_n_episodes)
    ground_state = np.zeros(total_n_episodes)
    remaining_sydromes = np.zeros(total_n_episodes)
    logical_errors = np.zeros(total_n_episodes)

    # initialize values to be accumulated across al episodes and steps
    syndromes_annihilated = 0
    syndromes_created = 0
    q_value_aggregation = np.zeros(total_n_episodes)
    q_value_diff_aggregation = np.zeros(total_n_episodes)
    q_value_certainty_aggregation = np.zeros(total_n_episodes)
    correct_actions_aggregation = np.zeros(num_of_user_episodes)

    # count how often an environment has finished and the number of steps it took
    env_done = np.zeros((total_n_episodes, 2))

    # # counts up to max_num_of_steps
    global_episode_steps = 0

    # initialize containers for evaluation metrics
    actions_in_one_episode = np.zeros((total_n_episodes, max_num_of_steps))

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

    if verbosity >= 5:
        all_q_values = np.zeros(
            (max_num_of_steps, total_n_episodes, 3 * code_size ** 2 + 1)
        )
    else:
        all_q_values = None

    # iterate through different episodes/environments
    # and handle them in parallel
    while global_episode_steps < max_num_of_steps:
        global_episode_steps += 1
        steps_per_episode += 1

        # let the agent do its work
        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        actions, q_values = select_actions(
            torch_states, model, code_size, epsilon=epsilon
        )

        if verbosity >= 5:
            all_q_values[global_episode_steps - 1, :, :] = q_values

        # revert action back to q value index fo later use
        q_value_indices = np.array(
            [
                action_to_q_value_index(actions[i], code_size)
                for i in range(total_n_episodes)
            ]
        )

        # check user-defined / expected actions
        # in user episodes
        correct_actions_all = np.array(
            [
                check_repeating_action(
                    actions[i],
                    expected_actions_per_episode[i - num_of_random_episodes],
                    len(expected_actions_per_episode[i - num_of_random_episodes]),
                )
                for i in range(
                    total_n_episodes - num_of_user_episodes, total_n_episodes
                )
            ]
        )
        correct_actions_aggregation += correct_actions_all

        # the predefined theoretical q values for user episodes
        # are only valid in the very first step,
        # and of course only in the user episodes
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

        # get info about max q values
        actions_in_one_episode[:, global_episode_steps - 1] = q_value_indices
        # gives the indices of the highest values for each row
        top_idx = np.argpartition(q_values, (-2, -1), axis=1)[:, -2:]
        highest_q_values = q_values[np.arange(q_values.shape[0])[:, None], top_idx]
        q_value = highest_q_values[:, -1]
        second_q_value = highest_q_values[:, -2]

        q_value_aggregation += q_value
        q_value_diff_aggregation += q_value - theoretical_q_values
        q_value_certainty_aggregation += q_value - second_q_value

        # apply the chosen action
        next_states, _, terminals, _ = env_set.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )

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

        # if we have a non-terminal episode and no net syndrome
        # creation or annihilation, at least the state must have changed
        # by applying an action.
        # Otherwise something went really wrong here
        non_terminal_episodes = np.where(actions[:, -1] != TERMINAL_ACTION)[0]
        for j in non_terminal_episodes:
            if n_annihilated_syndromes[j] == 0 and n_created_syndromes[j] == 0:
                # make sure that if a qubit changing action was chosen,
                # the syndrome state changes from one step to the next
                assert not np.all(
                    states[j] == next_states[j]
                ), f"{states[j]=}, \n{next_states[j]=}, \n{actions[0]=}"

        syndromes_annihilated += n_annihilated_syndromes.sum()
        syndromes_created += n_created_syndromes.sum()

        # Handle terminal episodes:
        # At this point, it would make more sense to talk about environments
        # rather than separate episodes.
        # In each environment with a terminal episode, we start a new one
        # and obtain its stats.
        # Its stats about number of steps are kept separately in env_done[:, 1]
        env_done[:, 0] += terminals
        if np.any(terminals):
            indices = np.argwhere(terminals).flatten()

            reset_states = env_set.reset_terminal_environments(indices)
            next_states[indices] = reset_states[indices]
            env_done[indices, 1] += steps_per_episode[indices]
            steps_per_episode[indices] = 0

            # check if calling a terminal action was a good choice
            for i in indices:
                _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                    env_set.environments[i].actual_errors,
                    env_set.environments[i].actions,
                    env_set.environments[i].vertex_mask,
                    env_set.environments[i].plaquette_mask,
                )

                ground_state[i] += _ground_state
                remaining_sydromes[i] += n_syndromes
                logical_errors[i] += n_loops

        states = next_states
        env_set.states = deepcopy(states)

    # end while; step through episode

    # check all states after the time limit
    # check for remaining syndromes etc.
    for i in range(total_n_episodes):
        unique, counts = np.unique(actions_in_one_episode[i], return_counts=True)
        most_common_qvalue_idx = np.argmax(counts)
        most_common_qvalue = unique[most_common_qvalue_idx]
        common_actions[i] = most_common_qvalue

        if terminals[i]:
            continue

        _, _ground_state, (n_syndromes, n_loops) = check_final_state(
            env_set.environments[i].actual_errors,
            env_set.environments[i].actions,
            env_set.environments[i].vertex_mask,
            env_set.environments[i].plaquette_mask,
        )

        ground_state[i] += _ground_state
        remaining_sydromes[i] += n_syndromes
        logical_errors[i] += n_loops
    # end for; loop over all episodes one last time

    # account for the case where environments might have finished more often than once
    avg_number_of_steps = average_episode_metric(
        steps_per_episode, total_n_episodes, env_done[:, 0], env_done[:, 1]
    )
    n_chose_correct_action_per_episode = np.mean(
        correct_actions_aggregation / num_of_user_episodes
    )
    n_ground_state = average_episode_metric(
        ground_state, total_n_episodes, env_done[:, 0]
    )
    n_remaining_syndromes = average_episode_metric(
        remaining_sydromes, total_n_episodes, env_done[:, 0]
    )
    n_logical_errors = average_episode_metric(
        logical_errors, total_n_episodes, env_done[:, 0]
    )
    n_terminals = average_episode_metric(terminals, total_n_episodes, env_done[:, 0])

    # normalize stats about syndrome creation
    # we want to have the probability for the agent
    # to create or destroy a syndrome in each step
    syndromes_normalization = syndromes_annihilated + syndromes_created
    syndromes_created /= syndromes_normalization
    syndromes_annihilated /= syndromes_normalization

    # q_value_aggregation contains info of every step in every episode
    mean_q_value = np.mean(q_value_aggregation) / max_num_of_steps
    mean_q_value_diff = np.mean(q_value_diff_aggregation) / max_num_of_steps
    std_q_value = np.std(q_value_aggregation) / max_num_of_steps
    q_value_certainty = np.mean(q_value_certainty_aggregation) / max_num_of_steps

    # An untrained network seems to choose a certain action lots of times
    # within an episode.
    # Make sure that it's at least a different action for different
    # state initializations.
    unique_actions = np.unique(common_actions)
    if not len(unique_actions) > 1:
        logger.debug(f"{common_actions=}, {common_actions.shape=}")
        logger.warning(
            "Warning! Only one action was chosen in all episodes. "
            f"Most common action index: {unique_actions}"
        )
    if verbosity >= 5:
        all_q_values = all_q_values.flatten()
    return (
        {
            RESULT_KEY_EPISODE: {
                "terminals_per_env": n_terminals,
                "ground_state_per_env": n_ground_state,
                "logical_errors_per_episode": n_logical_errors,
                "correct_actions_per_episode": n_chose_correct_action_per_episode,
                "remaining_syndromes_per_episode": n_remaining_syndromes,
            },
            RESULT_KEY_STEP: {
                "syndromes_annihilated_per_step": syndromes_annihilated,
                "syndromes_created_per_step": syndromes_created,
            },
            RESULT_KEY_P_ERR: {
                "avg_number_of_steps": avg_number_of_steps,
                "mean_q_value": mean_q_value,
                "std_q_value": std_q_value,
                "mean_q_value_difference": mean_q_value_diff,
                "q_value_certainty": q_value_certainty,
            },
        },
        {RESULT_KEY_HISTOGRAM_Q_VALUES: all_q_values},
    )


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


def count_spikes_np(arr, verbosity=0):
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
    global_steps += 1

    is_active = np.argwhere(1 - terminals).flatten()
    steps_per_episode[is_active] += 1

    assert len(states.shape) == 4  # need batch, stack_depth, d+1, d+1
    torch_states = torch.tensor(states[is_active], dtype=torch.float32).to(device)

    return global_steps, torch_states, is_active, steps_per_episode


def reset_local_actions_and_qvalues(terminal_actions, empty_q_values):
    actions = terminal_actions
    q_values = empty_q_values

    return actions, q_values


def get_two_highest_q_values(q_values):
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
    terminal_q_value
):
    q_value_aggregation += q_value
    q_value_diff_aggregation += q_value - theoretical_q_values
    q_value_certainty_aggregation += q_value - second_q_value
    terminal_q_value_aggregation += terminal_q_value

    return q_value_aggregation, q_value_diff_aggregation, q_value_certainty_aggregation, terminal_q_value_aggregation


def calc_theoretical_q_value(
    is_user_episode,
    steps_per_episode,
    theoretical_q_values,
    states,
    is_active,
    discount_factor_gamma,
    discount_intermediate_reward,
):
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
    num_raises = len(np.argwhere(np.diff(arr) > 0))
    return 2 * num_raises / len(arr)


def get_energy_stats(energies):
    energies = energies[energies > OUT_OF_RANGE]
    energy_spikes = count_spikes_np(energies)
    energy_raises = count_array_raises(energies)
    energy_final = energies[-1]
    energy_difference = energy_final - energies[0]

    return energy_spikes, energy_raises, energy_final, energy_difference


def get_intermediate_reward_stats(inter_rewards):
    inter_rewards = inter_rewards[inter_rewards > OUT_OF_RANGE]
    inter_rew_spikes = count_spikes_np(inter_rewards)
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
