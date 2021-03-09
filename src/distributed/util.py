import torch
import numpy as np
import random
from typing import List, Tuple, Union
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION


def incremental_mean(val, mu, n):
    return mu + (val - mu) / (n)


def select_action(
    state, model, system_size, num_actions_per_qubit=3, epsilon=0.0, device=None
):
    """
    Select an action according to an ε-greedy policy based on the
    provided neural network model.
    """

    model.eval()

    policy_net_output = None
    q_values = None
    with torch.no_grad():
        policy_net_output = model(state)
        q_values = np.array(policy_net_output.cpu())

    rand = random.random()

    # choose random action
    if rand < epsilon:
        q_value_probabilities = torch.softmax(q_values).detach().numpy()

        idx = np.random.choice(range(len(q_values[0])), p=q_value_probabilities)

    # choose deterministic, purely-greedy action
    else:
        idx = np.argmax(q_values[0])

    action = q_value_index_to_action(
        idx, system_size, num_actions_per_qubit=num_actions_per_qubit
    )

    return action, q_values


def action_to_q_value_index(
    action: Union[Tuple, List], system_size: int, num_actions_per_qubit: int = 3
) -> int:
    """
    Map an action, with its x- and y-coordinates and chosen operator,
    to the correct index in the q-value array.
    The q-value array should contain num_actions_per_qubit * system_size**2 + 1
    entries.
    The entries of q-value arrays are to be thought of
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
    actor = q_value_index % num_actions_per_qubit
    operator = actor + 1
    grid_index_group = (q_value_index - actor) // num_actions_per_qubit
    x_coord = grid_index_group % system_size
    y_coord = grid_index_group // system_size

    return (x_coord, y_coord, operator)


if __name__ == "__main__":
    for y in range(5):
        for x in range(5):
            for ac in (1, 2, 3):
                idx = action_to_q_value_index((x, y, ac), 5)
                print(f"{x=}, {y=}, {ac=}, {idx=}")

                action = q_value_index_to_action(idx, 5)
                assert action == (x, y, ac)

    for i in range(3 * 5 * 5 + 1):
        action = q_value_index_to_action(i, 5)
        x, y, ac = action
        print(f"{x=}, {y=}, {ac=}, {i=}")

        idx = action_to_q_value_index(action, 5)
        assert idx == i
