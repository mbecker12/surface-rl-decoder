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
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import (
    SOLVED_EPISODE_REWARD,
    SYNDROME_DIFF_REWARD,
    TERMINAL_ACTION,
    check_final_state,
    check_repeating_action,
    compute_layer_diff,
    create_syndrome_output_stack,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
def run_evaluation_in_batches(
    model,
    env_string,
    device,
    num_of_random_episodes=8,
    num_of_user_episodes=8,
    epsilon=0.0,
    max_num_of_steps=50,
    plot_one_episode=True,
    discount_factor_gamma=0.9,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.75,
    punish_repeating_actions=0,
    p_err=0.0,
    p_msmt=0.0,
):
    # TODO docstring

    model.eval()
    # initialize environments for different episodes
    total_n_episodes = num_of_random_episodes + num_of_user_episodes
    if env_string is None or env_string == "":
        env_string = SurfaceCode()

    print(f"{env_string.p_error=}")
    env_set = EnvironmentSet(env_string, total_n_episodes)
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
    syndromes_annihilated = 0
    syndromes_created = 0
    mean_q_value = 0
    mean_q_value_diff = 0
    q_value_aggregation = np.zeros(total_n_episodes)
    q_value_diff_aggregation = np.zeros(total_n_episodes)
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

    # TODO: comment code
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

    torch_states = torch.tensor(states, dtype=torch.float32).to(device)
    # TODO: remove asserts after testing on GPU
    assert torch_states.shape == (
        total_n_episodes,
        stack_depth,
        syndrome_size,
        syndrome_size,
    ), f"{torch_states.shape=}, {total_n_episodes=}, {stack_depth=}, {code_size=}"

    # TODO: make sure states are truly random
    count_same = 0
    for i in range(total_n_episodes):
        for j in range(i + 1, total_n_episodes):
            if torch.all(torch_states[i] == torch_states[j]):
                count_same += 1
    # TODO: remove asserts after testing on GPU
    assert count_same < total_n_episodes / 4, count_same

    while global_episode_steps < max_num_of_steps:
        global_episode_steps += 1
        steps_per_episode += 1

        # let the agent do its work
        actions, q_values = select_actions(
            torch_states, model, code_size, epsilon=epsilon
        )

        # loop through all episodes to obtain the current state
        # and get some information

        q_value_indices = np.array(
            [
                action_to_q_value_index(actions[i], code_size)
                for i in range(total_n_episodes)
            ]
        )

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
        # TODO: remove asserts after testing on GPU
        assert correct_actions_all.shape == (num_of_user_episodes,)

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

        actions_in_one_episode[:, global_episode_steps - 1] = q_value_indices
        q_value = np.take(q_values, q_value_indices)
        q_value_aggregation += q_value
        q_value_diff_aggregation += q_value - theoretical_q_values

        # TODO: remove asserts after testing on GPU
        assert q_value.shape == (total_n_episodes,), f"{q_value.shape=}"
        assert q_value_aggregation.shape == (
            total_n_episodes,
        ), f"{q_value_aggregation.shape=}"
        assert q_value_diff_aggregation.shape == (
            total_n_episodes,
        ), f"{q_value_diff_aggregation.shape=}"

        # apply the chosen action
        next_states, _, terminals, _ = env_set.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )

        diffs_all = np.array(
            [
                compute_layer_diff(states[i], next_states[i], stack_depth)
                for i in range(total_n_episodes)
            ]
        )

        # TODO: remove asserts after testing on GPU
        assert diffs_all.shape == (total_n_episodes, stack_depth)
        n_annihilated_syndromes = np.sum(diffs_all > 0, axis=1)
        assert n_annihilated_syndromes.shape == (total_n_episodes,)
        n_created_syndromes = np.sum(diffs_all < 0, axis=1)
        assert n_created_syndromes.shape == (total_n_episodes,)

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

        env_done[:, 0] += terminals
        if np.any(terminals):
            indices = np.argwhere(terminals).flatten()

            reset_states = env_set.reset_terminal_environments(indices)
            next_states[indices] = reset_states[indices]
            env_done[indices, 1] += steps_per_episode[indices]
            steps_per_episode[indices] = 0

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

    # end while; step through episode

    for i in range(total_n_episodes):
        unique, counts = np.unique(actions_in_one_episode[i], return_counts=True)
        most_common_qvalue_idx = np.argmax(counts)
        most_common_qvalue = unique[most_common_qvalue_idx]
        common_actions[i] = most_common_qvalue

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

    syndromes_normalization = syndromes_annihilated + syndromes_created
    syndromes_created /= syndromes_normalization
    syndromes_annihilated /= syndromes_normalization

    # q_value_aggregation contains info of every step in every episode
    mean_q_value = np.mean(q_value_aggregation) / max_num_of_steps
    mean_q_value_diff = np.mean(q_value_diff_aggregation) / max_num_of_steps

    unique_actions = np.unique(common_actions)
    if not len(unique_actions) > 1:
        logger.info(f"{common_actions=}, {common_actions.shape=}")
        logger.warning("Warning! Only one action was chosen in all episodes.")

    return (
        (
            n_terminals,
            n_ground_state,
            n_logical_errors,
            n_chose_correct_action_per_episode,
        ),
        (syndromes_annihilated, syndromes_created),
        (avg_number_of_steps, mean_q_value, mean_q_value_diff, n_remaining_syndromes),
    )


def create_user_eval_state(
    env: SurfaceCode,
    idx_episode: int,
    discount_factor_gamma=0.9,
    discount_intermediate_reward=0.75,
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
    env.state = create_syndrome_output_stack(
        env.qubits, env.vertex_mask, env.plaquette_mask
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
    n_syndromes_last_layer = np.sum(state[-1])
    # assume that syndromes (in the final layer) always come in pairs
    # this disregards edge qubits, where one qubit will only cause
    # one syndrome
    # Besides, this ignores the fact that Y errors cause more syndromes
    n_required_actions = np.ceil(n_syndromes_last_layer / 2)
    n_syndromes_total = np.sum(state)

    avg_syndrome_depth = (
        int(n_syndromes_total / n_syndromes_last_layer)
        if n_syndromes_last_layer > 0
        else 0
    )
    one_minus_inter_rew_discount_inv = 1.0 / (1.0 - discount_inter_reward)
    one_minus_gamma_inv = 1.0 / (1.0 - gamma)

    inter_reward = (
        2
        * SYNDROME_DIFF_REWARD
        * (1.0 - discount_inter_reward ** avg_syndrome_depth)
        * one_minus_inter_rew_discount_inv
    )

    final_reward = (
        inter_reward * (1.0 - gamma ** n_required_actions) * one_minus_gamma_inv
    )
    final_reward += SOLVED_EPISODE_REWARD * gamma ** n_required_actions
    return final_reward


def average_episode_metric(
    value_array, n_environments, finished_array, extra_values_array: List = [0, 0]
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
    # TODO: check linter hint on dangerous default value.
    # Actually, we're not changing the argument, only using the sum of it,
    # so it should be fine
    if np.any(finished_array):
        denominator_inv = 1.0 / (n_environments + np.sum(finished_array))
        avg_value = (np.sum(value_array) + np.sum(extra_values_array)) * denominator_inv
    else:
        avg_value = np.mean(value_array)

    return avg_value


if __name__ == "__main__":
    from distributed.dummy_agent import DummyModel

    config_ = {
        "layer1_size": 512,
        "layer2_size": 512,
        "code_size": 5,
        "syndrome_size": 6,
        "stack_depth": 8,
        "num_actions_per_qubit": 3,
    }
    model_ = DummyModel(config=config_)

    device_ = torch.device("cpu")
    from time import time

    start_time = time()
    for _ in range(3):
        eval_metrics = run_evaluation_in_batches(
            model_, "", device_, p_err=0.01, p_msmt=0.0
        )

        print(f"{eval_metrics=}")

    end_time = time()
    print(f"Time elapsed: {end_time - start_time}")
