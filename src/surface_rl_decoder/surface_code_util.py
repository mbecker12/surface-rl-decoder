"""
Utility functions for the surface code environment
"""
import numpy as np

TERMINAL_ACTION = 4

# Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3
RULE_TABLE = np.array(
    ([0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]), dtype=np.uint8
)

# reward scores
NON_TRIVIAL_LOOP_REWARD = -37
SYNDROME_LEFT_REWARD = -10
SOLVED_EPISODE_REWARD = 200
SYNDROME_DIFF_REWARD = 1
REPEATING_ACTION_REWARD = -2


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

    # look for uncorrected syndromes
    if (n_syndromes := final_state[-1].astype(np.uint8).sum()) != 0:
        return final_state, False, (n_syndromes, 0)

    # no syndromes left
    # check for non-trivial loops
    z_errors = (final_qubit_configuration[-1] == 3).astype(np.uint8)
    y_errors = (final_qubit_configuration[-1] == 2).astype(np.uint8)
    x_errors = (final_qubit_configuration[-1] == 1).astype(np.uint8)

    x_matrix = x_errors + y_errors
    z_matrix = y_errors + z_errors

    x_loops = np.sum(np.sum(x_matrix, axis=0))
    z_loops = np.sum(np.sum(z_matrix != 0, axis=0))
    n_loops = x_loops % 2 + z_loops % 2

    is_ground_state = True

    if n_loops > 0:
        is_ground_state = False

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
    vertex_mask: (1, d+1, d+1) or (h, d+1, d+1) logical mask denoting locations
        of vertex ancillaries in the syndrome encoding
    plaquette_mask: (1, d+1, d+1) or (h, d+1, d+1) logical mask denoting locations
        of plaquette ancillaries in the syndrome encoding

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

    # X operators = shaded = vertex = checks for Z errors
    syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * plaquette_mask[0]
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * plaquette_mask[0]

    # Z operators = blank = plaquette = checks for X errors
    syndrome += (z + z_shifted_up + z_shifted_left + z_shifted_ul) * vertex_mask[0]
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * vertex_mask[0]

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
    vertex_mask: (h, d+1, d+1) logical mask denoting locations
        of vertex ancillaries in the syndrome encoding
    plaquette_mask: (h, d+1, d+1) logical mask denoting locations
        of plaquette ancillaries in the syndrome encoding


    Returns
    =======
    syndrome: (h, d+1, d+1) array embedding vertices and plaquettes
    """
    # pad with (
    #   (nothing along time axis),
    #   (one row above, zero rows below),
    #   (one row to the left, zero rows to the right)
    # )
    padded_qubits = np.pad(
        qubits, ((0, 0), (1, 0), (1, 0)), "constant", constant_values=0
    )

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

    # X operators = shaded = vertex = checks for Z errors
    syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * plaquette_mask
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * plaquette_mask

    # Z operators = blank = plaquette = checks for X errors
    syndrome += (z + z_shifted_up + z_shifted_left + z_shifted_ul) * vertex_mask
    syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * vertex_mask

    assert syndrome.shape == (
        padded_qubits.shape[0],
        padded_qubits.shape[1],
        padded_qubits.shape[2],
    ), (syndrome.shape, qubits.shape)

    syndrome = (
        syndrome % 2
    )  # we can only measure parity, hence only odd number of errors per syndrome
    return syndrome


def compute_intermediate_reward(
    state, next_state, stack_depth, discount_factor=0.75, annealing_factor=1.0
):
    """
    Calculate an intermediate reward based on the number of created/annihilated syndromes
    in the syndrome stack.
    This looks throughout the whole stack and looks for differences in
    number of syndromes in each layer. The earlier the layer, the more discounted
    its contribution to the reward will be.

    Parameters
    ==========
    state: (h, d+1, d+1) current syndrome state
    next_state: (h, d+1, d+1) subsequent syndrome state
    stack_depth: number of layers in the syndrome stack, a.k.a. h
    discount_factor: (optional) discount factor determining how much
        early layers should be discounted when calculating the intermediate reward
    annealing_factor: (optional) variable that should decrease over time during
        a training run to decrease the effect of the intermediate reward

    Returns
    =======
    intermediate_reward: (float) reward for annihilating/creating a syndrome across the stack
    """

    diffs = compute_layer_diff(state, next_state, stack_depth)
    layer_exponents = np.arange(stack_depth - 1, -1, -1)
    layer_rewards = (
        annealing_factor
        * SYNDROME_DIFF_REWARD
        * diffs
        * np.power(discount_factor, layer_exponents)
    )

    intermediate_reward = np.sum(layer_rewards)
    return intermediate_reward


def compute_layer_diff(state, next_state, stack_depth):
    """
    Utility function to compute the layerwise difference
    in the number of syndrome measurements between a state and
    its subsequent state.

    diffs = state - next_state
    Hence, a positive number means a decrease in syndrome measurements.

    Parameters
    ==========
    state: (h, d+1, d+1) current syndrome state
    next_state: (h, d+1, d+1) subsequent syndrome state
    stack_depth: number of layers in the syndrome stack, a.k.a. h

    Return
    ======
    diffs: (h,) difference in number of
        syndrome measurements between subsequent layers
    """

    state_sums = np.sum(np.sum(state, axis=2), axis=1)
    next_state_sums = np.sum(np.sum(next_state, axis=2), axis=1)
    diffs = state_sums - next_state_sums
    assert diffs.shape == (stack_depth,), diffs.shape
    assert diffs.dtype in (int, float, np.uint64), diffs.dtype

    return diffs


def check_repeating_action(action, action_history, max_action_index):
    """
    Check the action history of an environment if the proposed action
    has been executed before.

    Parameters
    ==========
    action: Tuple, shape: (3, ), (x-coord, y-coord, operator) action to perform on a qubit
    action_history: array that stores the previously executed action-tuples
    max_action_index: the index up to which the action history is filled

    Returns
    =======
    n_repeating_actions: the number of how often action has already occured in the action histoy
    """
    n_repeating_actions = sum(
        [np.all(action == action_history[i]) for i in range(max_action_index)]
    )
    return n_repeating_actions
