import torch
import numpy as np
import random
from typing import List, Tuple, Union
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION


def incremental_mean(val, mu, n):
    return mu + (val - mu) / (n)


def select_actions(state, model, system_size, num_actions_per_qubit=3, epsilon=0.0):
    """
    Select actions batch-wise according to an ε-greedy policy based on the
    provided neural network model.

    Parameters
    ==========
    state: torch.tensor, batch of stacks of states,
        shape: (batch_size, stack_depth, system_size, system_size)
    model: the neural network model of choice
    num_actions_per_qubit: (optional) number of possible operators on one qubit,
        default is 3, for Pauli-X, -Y, or -Z.
    epsilon: (float) probability to choose a random action

    Returns
    =======
    actions: (array-like) shape: (batch_size, 3); chosen action for each batch
    q_values: (array-like), shape: (batch_size, num_actions_per_qubit * d**2 + 1)
        q_values for each action in the given input state.
    """

    model.eval()

    policy_net_output = None
    q_values = None
    with torch.no_grad():
        policy_net_output = model(state)
        q_values = np.array(policy_net_output.cpu())

    batch_size = q_values.shape[0]

    # choose completely greedy action first
    # for now, choose only the q-value-index
    # the actual action is chosen at the bottom of this function
    q_value_index = np.argmax(q_values, axis=1)

    # choose random action where it applies
    rand = np.random.random_sample(batch_size)
    non_greedy_mask = rand < epsilon  # where true: choose a random action
    non_greedy_indices = np.where(non_greedy_mask)[0]

    # generate probabilities
    q_value_probabilities = (
        torch.softmax(policy_net_output[non_greedy_indices], dim=1, dtype=torch.float32)
        .detach()
        .numpy()
    )

    assert q_value_probabilities.shape[0] == len(non_greedy_indices), len(
        non_greedy_indices
    )

    # assure that probabilities add to 1
    for j, _ in enumerate(non_greedy_indices):
        assert (
            0.999 <= q_value_probabilities[j].sum() <= 1.00001
        ), q_value_probabilities[j].sum()

    # overwrite the chosen q-value index at the places where a random action
    # should be chosen
    q_value_index[non_greedy_indices] = np.array(
        [
            np.random.choice(range(len(q_values[0])), p=q_value_probabilities[j])
            for j, _ in enumerate(non_greedy_indices)
        ]
    )

    # finally choose the actual action based on the q-value index
    actions = np.array(
        [
            q_value_index_to_action(
                q_value_index[i],
                system_size,
                num_actions_per_qubit=num_actions_per_qubit,
            )
            for i in range(batch_size)
        ]
    )

    assert len(actions) == batch_size

    return actions, q_values


def action_to_q_value_index(
    action: Union[Tuple, List], system_size: int, num_actions_per_qubit: int = 3
) -> int:
    """
    Map an action, with its x- and y-coordinates and chosen operator,
    to the correct index in the q-value array.
    The q-value array should contain num_actions_per_qubit * system_size**2 + 1
    entries.
    The entries of q-value arrays are thought to be the actions in the following order:
        [
            (0,0,1), (0,0,2), (0,0,3), (1,0,1), (1,0,2), (1,0,3), (2,0,1), ...,
            (d-2, d-1, 3), (d-1, d-1, 1), (d-1, d-1, 2), (d-1, d-1, 3), (x, y, terminal)
        ]

    Parameters
    ==========
    action: (Tuple) (x-coordinate, y-coordinate, operator) of one action
        to be performed on the whole qubit stack
    system_size: code distance d, number of physical qubits per row/column
    num_actions_per_qubit: (optional) number of possible operators on one qubit,
        default is 3, for Pauli-X, -Y, or -Z.

    Returns
    =======
    index: Index from which to select from q-value-array.

    """
    x_coord, y_coord, operator = action[-3:]

    if 1 <= operator <= 3:
        assert x_coord >= 0, "qubit x coordinate must be between 0 and d-1"
        assert y_coord >= 0, "qubit y coordinate must be between 0 and d-1"
        index = (
            x_coord * num_actions_per_qubit
            + y_coord * system_size * num_actions_per_qubit
            + (operator - 1)
        )
    elif operator == TERMINAL_ACTION:
        index = num_actions_per_qubit * system_size * system_size
    else:
        raise Exception(
            f"Error! Operator {operator} on qubit ({x_coord}, {y_coord}) is not defined."
        )
    return index


def q_value_index_to_action(q_value_index, system_size, num_actions_per_qubit=3):
    """
    Map an index from a alid q-value array to the corresponding action,
    with its x- and y-coordinates and chosen operator.
    The q-value array should contain num_actions_per_qubit * system_size**2 + 1
    entries.
    The entries of q-value arrays are thought to be the actions in the following order:
        [
            (0,0,1), (0,0,2), (0,0,3), (1,0,1), (1,0,2), (1,0,3), (2,0,1), ...,
            (d-2, d-1, 3), (d-1, d-1, 1), (d-1, d-1, 2), (d-1, d-1, 3), (x, y, terminal)
        ]

    Parameters
    ==========
    q_value_index: (int) index pointing to the desired q-value in the q-value-array
    system_size: code distance d, number of physical qubits per row/column
    num_actions_per_qubit: (optional) number of possible operators on one qubit,
        default is 3, for Pauli-X, -Y, or -Z.

    Returns
    =======
    action: (Tuple) (x-coordinate, y-coordinate, operator) of one action
        to be performed on the whole qubit stack
    """
    # example, assuming system_size=5, actions_per_qubit=3
    # (example: index 22) -> action (2, 1, 2)
    # actor = 22 % 3 = 1
    # grid_index_group = (22 - 1) // 3 = 7
    # x = 7 % 5 = 2
    # y = 7 // 5 = 1
    if (
        q_value_index == num_actions_per_qubit * system_size * system_size
        or q_value_index == -1
    ):
        return (0, 0, TERMINAL_ACTION)
    elif (
        q_value_index < 0
        or q_value_index > num_actions_per_qubit * system_size * system_size
    ):
        raise Exception(
            f"Error! Index {q_value_index} is invalid for surface code with system size {system_size}."
        )

    actor = q_value_index % num_actions_per_qubit
    operator = actor + 1
    grid_index_group = (q_value_index - actor) // num_actions_per_qubit
    x_coord = grid_index_group % system_size
    y_coord = grid_index_group // system_size

    return (x_coord, y_coord, operator)


def assert_not_all_elements_equal(arr):
    """
    Helper function to make sure that not
    all the elements in a given array are the same.
    Only checks neighboring entries.
    """
    diff = np.diff(np.squeeze(arr))
    assert np.any(diff != 0), f"{arr=}, {diff=}"


def assert_not_all_states_equal(states_batch):
    """
    Helper function to make sure that not
    all the syndrome states in a given batch are the same.
    Only checks neighboring states.
    """
    assert states_batch is not None
    batch_size = len(states_batch)
    depth = states_batch.shape[1]

    count_same = 0
    total = 0

    for i in range(1, batch_size):
        prev_state = states_batch[i - 1]
        current_state = states_batch[i]

        assert prev_state is not None
        assert current_state is not None

        for h in range(depth):
            if isinstance(states_batch, torch.Tensor):
                if torch.all(prev_state[h] == current_state[h]):
                    count_same += 1

                total += 1
            elif isinstance(states_batch, np.ndarray):
                if np.all(prev_state[h] == current_state[h]):
                    count_same += 1

                total += 1
            else:
                raise Exception(
                    f"Error! Data type {type(states_batch).__name__} not supported."
                )

        similarity = count_same / total

    return similarity


def compute_priorities(actions, rewards, qvalues, qvalues_new, gamma, system_size):
    """
    Compute the absolute temporal difference (TD) value, to be used
    as priority for replay memory.

    TD_error = R + γ * Q_max(s(t+1), a) - Q(s(t), a)

    Using the TD error as the priority allows sampling events that cause
    a large loss (and hence a potentially large change in network weights)
    to be sampled more often.

    Parameters
    ==========
    actions: (n_environments, buffer_size, 3) batches of actions
    rewards: (n_environments, buffer_size) batches of rewards
    q values: (n_environments, buffer_size, 3 * d**2 + 1) batches of q values
    qvalues_new: (n_environments, buffer_size, 3 * d**2 + 1)
        look-ahead from saved transitions, the q values of the subsequent state
    gamma: (float) discount factor
    system_size: (int) code size, d

    Returns
    =======
    priorities: (n_environments, buffer_size) absolute TD error for each sample
    """
    qmax = np.amax(qvalues_new, axis=2)

    n_envs = qvalues.shape[0]
    n_bufs = qvalues.shape[1]

    selected_q_values = np.array(
        [
            [
                qvalues[env][buf][
                    action_to_q_value_index(actions[env][buf], system_size)
                ]
                for buf in range(n_bufs)
            ]
            for env in range(n_envs)
        ]
    )

    priorities = np.absolute(rewards + gamma * qmax - selected_q_values)

    return priorities
