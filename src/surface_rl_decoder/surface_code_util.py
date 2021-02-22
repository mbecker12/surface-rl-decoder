"""
Utility functions for the surface code environment
"""
import numpy as np

TERMINAL_ACTION = 4
MAX_ACTIONS = 256

# Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3
RULE_TABLE = np.array(
    ([0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]), dtype=np.uint8
)


def check_final_state(actual_errors, actions):
    """
    Returns the final state, i.e. initial qubit configuration
    with all suggested actions executed on it.
    Returns if this final state is in the ground state.

    Parameters
    ==========
    actual_errors: (h, d, d) array of initial qubit stack without measurement errors
    actions: (max_actions, 3) array of stored action history

    Returns
    =======
    final_state: (h, d, d) array of final state,
        i.e. all proposed actions executed on initial qubit state
    is_ground_state: (bool) shows whether final_state is in the ground state or not
    """
    # check for trivial loops
    # check for logical operation
    # check if still errors left

    final_state = perform_all_actions(actual_errors, actions)

    z_errors = (final_state[-1] == 3).astype(np.uint8)
    y_errors = (final_state[-1] == 2).astype(np.uint8)
    x_errors = (final_state[-1] == 1).astype(np.uint8)

    x_matrix = x_errors + y_errors
    z_matrix = y_errors + z_errors

    x_loops = np.sum(np.sum(x_matrix, axis=0))
    z_loops = np.sum(np.sum(z_matrix, axis=0))

    is_ground_state = True

    if x_loops % 2 == 1:
        is_ground_state = False
    elif z_loops % 2 == 1:
        is_ground_state = False

    return final_state, is_ground_state


def perform_all_actions(qubits, actions):
    """
    Perform all actions in the action history

    Parameters
    ==========
    qubits: (h, d, d) array of qubit stack,
        could for example be the initial qubit arrangement
        without any measurement errors
    actions: (max_actions, 3) list of all actions suggested by the agent
        contains a list of actions in the form (x, y, operator)

    Returns
    =======
    qubits: (h, d, d) array of qubit stack on which all actions
        that were suggested by the agent have been performed.
        Therefore, in the optimal case, there should be no
        physical errors left.
    """
    for action in actions:
        if action[-1] == TERMINAL_ACTION:
            return qubits
        qubits = perform_action(qubits, action)
    return qubits


def perform_action(qubits, action):
    """
    Perform one action throughout the whole stack.

    Parameters
    ==========
    qubits: (h, d, d) qubit stack
    action: tuple containing (None, x-coordinate, y-coordinate, pauli operator),
        defining x- & y-coordinates and operator type

    Returns
    =======
    qubits: (h, d, d) qubit stack on which one operation
        (defined in function argument action) has been performed
        on all time slices in the stack.
    """
    row, col, add_operator = action[-3:]
    if add_operator == TERMINAL_ACTION:
        raise Exception("Error! Attempting to execute terminal operation.")
    old_operator = qubits[:, row, col]
    new_operator = [RULE_TABLE[old_op, add_operator] for old_op in old_operator]
    qubits[:, row, col] = new_operator
    return qubits


def is_terminal(action):
    """
    Look at the agent's action.
    If the agent chose the 'terminal' action, the episode
    is supposed to be terminated.

    Parameters
    ==========
    action: tuple containing (None, x-coordinate, y-coordinate, pauli operator),
        defining x- & y-coordinates and operator type

    Returns
    =======
    is_terminal (bool)
    """
    return action[-1] == TERMINAL_ACTION


def copy_array_values(source_array):
    """
    Copy an array to a new one, value by value
    to avoid having to use deepcopy
    and to avoid potential memory leaks.

    Parameters
    ==========
    source_array: 3-dimensional array to copy values from
    dimension: shape of source_array

    Returns
    =======
    target_array: New array with same values and shape as source_array
    """
    dimension = source_array.shape
    assert len(dimension) == 3

    target_array = np.array(
        [
            [
                [source_array[h, i, j] for j in range(dimension[2])]
                for i in range(dimension[1])
            ]
            for h in range(dimension[0])
        ]
    )
    return target_array
