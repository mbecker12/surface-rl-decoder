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

# reward scores
NON_TRIVIAL_LOOP_REWARD = -1
SYNDROME_LEFT_REWARD = -5
SOLVED_EPISODE_REWARD = 100

# TODO: docstrings for create_syndrome functions


def check_final_state(actual_errors, actions, vertex_mask, plaquette_mask):
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
    (n_syndromes, n_loops): (tuple) counting the number of syndromes and/or loops
        remaining in the final layer after all corrections have been performed
    """

    final_qubit_configuration = perform_all_actions(actual_errors, actions)
    final_state = create_syndrome_output(
        final_qubit_configuration[-1],
        vertex_mask=vertex_mask,
        plaquette_mask=plaquette_mask,
    )
    final_state = final_state.reshape(1, final_state.shape[-2], final_state.shape[-1])

    print(f"Check for syndromes")
    # look for uncorrected syndromes
    print(final_state[-1])
    if (n_syndromes := final_state[-1].astype(np.uint8).sum()) != 0:
        print(f"Have uncorrected syndromes")
        # print(f"{n_syndromes=}")
        # print(f"{final_state[-1]=}")
        return final_state, False, (n_syndromes, 0)

    # no syndromes left
    # check for non-trivial loops
    print(f"No syndromes")
    print(f"Check for non-trivial loops")
    z_errors = (final_qubit_configuration[-1] == 3).astype(np.uint8)
    y_errors = (final_qubit_configuration[-1] == 2).astype(np.uint8)
    x_errors = (final_qubit_configuration[-1] == 1).astype(np.uint8)

    x_matrix = x_errors + y_errors
    z_matrix = y_errors + z_errors

    x_loops = np.sum(np.sum(x_matrix, axis=0))
    z_loops = np.sum(np.sum(z_matrix != 0, axis=0))
    n_loops = x_loops + z_loops

    is_ground_state = True

    if x_loops % 2 == 1:
        is_ground_state = False
    elif z_loops % 2 == 1:
        is_ground_state = False

    print(f"{n_loops=}")
    return final_state, is_ground_state, (0, n_loops)


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


def create_syndrome_output(qubits, vertex_mask, plaquette_mask):
    """
    Infer the true syndrome output (w/o measurement errors)
    from the qubit matrix.
    Perform this for one slice.

    Parameters
    ==========
    qubits: (d, d) array containing the net operation performed on each qubit

    Returns
    =======
    syndrome: (d+1, d+1) array embedding vertices and plaquettes
    """

    # make sure it is only one slice
    if len(qubits.shape) != 2:
        if len(qubits.shape) == 3:
            assert qubits.shape[0] == 1, qubits.shape

    # pad with ((one row above, zero rows below), (one row to the left, zero rows to the right))
    padded_qubits = np.pad(qubits, ((1, 0), (1, 0)), "constant", constant_values=0)

    # pylint: disable=invalid-name
    x = (padded_qubits == 1).astype(np.uint8)
    y = (padded_qubits == 2).astype(np.uint8)
    z = (padded_qubits == 3).astype(np.uint8)
    assert x.shape == padded_qubits.shape, x.shape
    assert y.shape == padded_qubits.shape, y.shape
    assert z.shape == padded_qubits.shape, z.shape

    x_shifted_left = np.roll(x, -1, axis=1)
    x_shifted_up = np.roll(x, -1, axis=0)
    x_shifted_ul = np.roll(x_shifted_up, -1, axis=1)  # shifted up and left

    z_shifted_left = np.roll(z, -1, axis=1)
    z_shifted_up = np.roll(z, -1, axis=0)
    z_shifted_ul = np.roll(z_shifted_up, -1, axis=1)

    y_shifted_left = np.roll(y, -1, axis=1)
    y_shifted_up = np.roll(y, -1, axis=0)
    y_shifted_ul = np.roll(y_shifted_up, -1, axis=1)

    # X = shaded = vertex
    syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * vertex_mask[0]
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * vertex_mask[0]

    # Z = blank = plaquette
    syndrome += (z + z_shifted_up + z_shifted_left + z_shifted_ul) * plaquette_mask[0]
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * plaquette_mask[0]

    assert syndrome.shape == (
        padded_qubits.shape[-2],
        padded_qubits.shape[-1],
    ), syndrome.shape

    syndrome = (
        syndrome % 2
    )  # we can only measure parity, hence only odd number of errors per syndrome
    return syndrome


def create_syndrome_output_stack(qubits, vertex_mask, plaquette_mask):
    """
    Infer the true syndrome output (w/o measurement errors)
    from the qubit matrix.

    d: code distance
    h: stack depth/height

    Parameters
    ==========
    qubits: (h, d, d) array containing the net operation performed on each qubit

    Returns
    =======
    syndrome: (h, d+1, d+1) array embedding vertices and plaquettes
    """
    # pad with (
    #   (nothing along time axis),
    #   (one row above, zero rows below),
    #   (one row to the left, zero rows to the right)
    # )
    padded_qubits = np.pad(qubits, ((0, 0), (1, 0), (1, 0)), "constant", constant_values=0)

    # pylint: disable=invalid-name
    x = (padded_qubits == 1).astype(np.uint8)
    y = (padded_qubits == 2).astype(np.uint8)
    z = (padded_qubits == 3).astype(np.uint8)
    assert x.shape == padded_qubits.shape
    assert y.shape == padded_qubits.shape
    assert z.shape == padded_qubits.shape

    x_shifted_left = np.roll(x, -1, axis=2)
    x_shifted_up = np.roll(x, -1, axis=1)
    x_shifted_ul = np.roll(x_shifted_up, -1, axis=2)  # shifted up and left

    z_shifted_left = np.roll(z, -1, axis=2)
    z_shifted_up = np.roll(z, -1, axis=1)
    z_shifted_ul = np.roll(z_shifted_up, -1, axis=2)

    y_shifted_left = np.roll(y, -1, axis=2)
    y_shifted_up = np.roll(y, -1, axis=1)
    y_shifted_ul = np.roll(y_shifted_up, -1, axis=2)

    # X = shaded = vertex
    syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * vertex_mask
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * vertex_mask

    # Z = blank = plaquette
    syndrome += (z + z_shifted_up + z_shifted_left + z_shifted_ul) * plaquette_mask
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * plaquette_mask

    assert syndrome.shape == (
        padded_qubits.shape[0],
        padded_qubits.shape[1],
        padded_qubits.shape[2],
    ), (syndrome.shape, qubits.shape)

    syndrome = (
        syndrome % 2
    )  # we can only measure parity, hence only odd number of errors per syndrome
    return syndrome