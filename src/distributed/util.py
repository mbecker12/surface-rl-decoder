"""
General utility functions for the distributed program setup.
Contains many of the actor utilities.
"""
from time import time
from typing import List, Tuple, Union
from numpy.lib.function_base import select
import torch
import numpy as np
from torch.tensor import Tensor
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION

# from iniparser import Config

# c = Config()
# _config = c.scan(".", True).read()
# config = c.config_rendered

# env_config = config.get("config").get("env")
# learner_config = config.get("config").get("learner")
# device = learner_config.get("device")
# d = int(env_config.get("size"))

# pylint: disable=not-callable
LOCAL_X_DELTA = torch.tensor([[0, 1], [1, 0]], dtype=torch.int8)
LOCAL_Y_DELTA = torch.tensor([[1, 1], [1, 1]], dtype=torch.int8)
LOCAL_Z_DELTA = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)

LOCAL_DELTAS = torch.stack([LOCAL_X_DELTA, LOCAL_Y_DELTA, LOCAL_Z_DELTA])

COORDINATE_SHIFT_1 = [-1, -1]
COORDINATE_SHIFT_2 = [-1, 0]
COORDINATE_SHIFT_3 = [0, -1]
COORDINATE_SHIFT_4 = [0, 0]
COORDINATE_SHIFTS = torch.tensor(
    [COORDINATE_SHIFT_1, COORDINATE_SHIFT_2, COORDINATE_SHIFT_3, COORDINATE_SHIFT_4],
    dtype=torch.int8,
)

from torch.nn.functional import pad as torch_pad


def incremental_mean(value, mean, num_elements):
    """
    Iteratively update the mean of a value.
    """
    return mean + (value - mean) / (num_elements)


def select_actions(state, model, code_size, num_actions_per_qubit=3, epsilon=0.0):
    """
    Select actions batch-wise according to an ε-greedy policy based on the
    provided neural network model.

    Parameters
    ==========
    state: torch.tensor, batch of stacks of states,
        shape: (batch_size, stack_depth, code_size, code_size)
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
        q_values_torch_cpu = policy_net_output.detach().cpu()
        q_values = np.array(q_values_torch_cpu)

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
        torch.softmax(
            q_values_torch_cpu[non_greedy_indices], dim=1, dtype=torch.float32
        )
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
                code_size,
                num_actions_per_qubit=num_actions_per_qubit,
            )
            for i in range(batch_size)
        ]
    )

    assert len(actions) == batch_size

    return actions, q_values


def action_to_q_value_index(
    action: Union[Tuple, List], code_size: int, num_actions_per_qubit: int = 3
) -> int:
    """
    Map an action, with its x- and y-coordinates and chosen operator,
    to the correct index in the q-value array.
    The q-value array should contain num_actions_per_qubit * code_size**2 + 1
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
    code_size: code distance d, number of physical qubits per row/column
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
            + y_coord * code_size * num_actions_per_qubit
            + (operator - 1)
        )
    elif operator == TERMINAL_ACTION:
        index = num_actions_per_qubit * code_size * code_size
    else:
        raise Exception(
            f"Error! Operator {operator} on qubit ({x_coord}, {y_coord}) is not defined."
        )
    return index


def q_value_index_to_action(q_value_index, code_size, num_actions_per_qubit=3):
    """
    Map an index from a alid q-value array to the corresponding action,
    with its x- and y-coordinates and chosen operator.
    The q-value array should contain num_actions_per_qubit * code_size**2 + 1
    entries.
    The entries of q-value arrays are thought to be the actions in the following order:
        [
            (0,0,1), (0,0,2), (0,0,3), (1,0,1), (1,0,2), (1,0,3), (2,0,1), ...,
            (d-2, d-1, 3), (d-1, d-1, 1), (d-1, d-1, 2), (d-1, d-1, 3), (x, y, terminal)
        ]

    Parameters
    ==========
    q_value_index: (int) index pointing to the desired q-value in the q-value-array
    code_size: code distance d, number of physical qubits per row/column
    num_actions_per_qubit: (optional) number of possible operators on one qubit,
        default is 3, for Pauli-X, -Y, or -Z.

    Returns
    =======
    action: (Tuple) (x-coordinate, y-coordinate, operator) of one action
        to be performed on the whole qubit stack
    """
    # example, assuming code_size=5, actions_per_qubit=3
    # (example: index 22) -> action (2, 1, 2)
    # actor = 22 % 3 = 1
    # grid_index_group = (22 - 1) // 3 = 7
    # x = 7 % 5 = 2
    # y = 7 // 5 = 1
    assert code_size is not None
    if q_value_index in (num_actions_per_qubit * code_size * code_size, -1):
        return (0, 0, TERMINAL_ACTION)

    if (
        q_value_index < 0
        or q_value_index > num_actions_per_qubit * code_size * code_size
    ):
        raise Exception(
            f"Error! Index {q_value_index} "
            "is invalid for surface code with system size {code_size}."
        )

    actor = q_value_index % num_actions_per_qubit
    operator = actor + 1
    grid_index_group = (q_value_index - actor) // num_actions_per_qubit
    x_coord = grid_index_group % code_size
    y_coord = grid_index_group // code_size

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

        for height in range(depth):
            if isinstance(states_batch, torch.Tensor):
                if torch.all(prev_state[height] == current_state[height]):
                    count_same += 1

                total += 1
            elif isinstance(states_batch, np.ndarray):
                if np.all(prev_state[height] == current_state[height]):
                    count_same += 1

                total += 1
            else:
                raise Exception(
                    f"Error! Data type {type(states_batch).__name__} not supported."
                )

        similarity = count_same / total

    return similarity


def compute_priorities(
    actions, rewards, qvalues, qvalues_new, gamma, code_size, rl_type="q"
):
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
    code_size: (int) code size, d

    Returns
    =======
    priorities: (n_environments, buffer_size) absolute TD error for each sample
    """
    qmax = np.amax(qvalues_new, axis=2)

    n_envs = qvalues.shape[0]
    n_bufs = qvalues.shape[1]

    if rl_type == "q":
        selected_q_values = np.array(
            [
                [
                    qvalues[env][buf][
                        action_to_q_value_index(actions[env][buf], code_size)
                    ]
                    for buf in range(n_bufs)
                ]
                for env in range(n_envs)
            ]
        )

        priorities = np.absolute(rewards + gamma * qmax - selected_q_values)
    elif rl_type == "v":
        priorities = np.absolute(
            rewards + gamma * qvalues_new.squeeze() - qvalues.squeeze()
        )

    return priorities


def anneal_factor(
    time_difference=None,
    timesteps=None,
    decay_factor=1.0,
    min_value=0.0,
    max_value=1.0,
    base_factor=1.0,
):
    """
    Compute a general time-dependent anneal factor.
    It can then be multiplied to a given hyperparameter to reduce its
    effect over time.
    Needs either a time difference or number of timesteps as input.

    Parameters
    ==========
    time_difference: absolute time that has passed since the program was started
    timesteps: number of time steps that have passed
    decay_factor: the rate of decay; this will be exponentiated by the time value
    min_value: lower bound of annealing effect
    min_value: upper bound of annealing effect
    base_factor: additional multiplicator, can be the actual hyperparameter
        to anneal

    Returns
    =======
    annealing_factor: multiplicator to adjust the effect of a hyperparameter
    """
    if time_difference is None:
        assert timesteps is not None

        annealing_factor = max(min_value, base_factor * decay_factor ** timesteps)
        annealing_factor = min(max_value, annealing_factor)
        return annealing_factor

    if timesteps is None:
        assert time_difference is not None

        annealing_factor = max(min_value, base_factor * decay_factor ** time_difference)
        annealing_factor = min(max_value, annealing_factor)
        return annealing_factor

    raise Exception(
        "Error! You need to either define the timesteps or the time_difference that has passed."
    )


def time_tb():
    """
    Get the current time in seconds.
    And see how tensorboard is able to handle that...
    But still it can't handle it properly... ¯\\_(ツ)_/¯
    """
    return int(time())


def select_actions_value_network(
    state_batch,
    model,
    code_size,
    stack_depth,
    combined_mask,
    coordinate_shifts,
    local_deltas,
    device,
    num_actions_per_qubit=3,
    epsilon=0.0,
):
    """
    Select actions batch-wise according to an ε-greedy policy based on the
    provided neural network model obeying the value network learning scheme.

    Parameters
    ==========
    state: torch.tensor, batch of stacks of states,
        shape: (batch_size, stack_depth, code_size, code_size)
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

    coordinate_shifts = format_torch(coordinate_shifts, device=device, dtype=torch.int8)
    combined_mask = format_torch(combined_mask, device=device, dtype=torch.int8)
    local_deltas = format_torch(local_deltas, device=device, dtype=torch.int8)
    state_batch = format_torch(state_batch, device=device, dtype=torch.uint8)

    batch_size = state_batch.shape[0]
    batch_selected_actions = [None] * batch_size
    batch_selected_values = [None] * batch_size

    with torch.no_grad():
        for i, state in enumerate(state_batch):
            # filter reasonable actions
            possible_actions = determine_possible_actions(
                state, code_size, coordinate_shifts=coordinate_shifts, device=device
            )
            l_actions = len(possible_actions) + 1

            # get operators corresponding to the above filtered actions
            operators = create_possible_operators(
                possible_actions,
                code_size + 1,
                stack_depth,
                combined_mask,
                local_deltas,
                max_l=l_actions,
            )

            stacked_state = torch.tile(
                state,
                (l_actions, 1, 1, 1),
            )

            # apply operators to the state to create successor states
            operators = operators.to(device)
            stacked_state = stacked_state.to(device)
            new_states = torch.logical_xor(stacked_state, operators)
            new_states = new_states.to(device=device, dtype=torch.float32)
            assert new_states.shape == (
                l_actions,
                stack_depth,
                code_size + 1,
                code_size + 1,
            ), new_states.shape

            policy_net_output = model(new_states)
            assert policy_net_output.shape == (l_actions, 1), policy_net_output.shape
            values_torch_cpu = policy_net_output.detach().cpu()
            assert values_torch_cpu.shape == (l_actions, 1), values_torch_cpu.shape
            values = np.array(values_torch_cpu)
            assert values.shape == (l_actions, 1), values.shape

            # TODO:
            # redo random action choosing
            # need to collect all successor states and values first

            rand = np.random.random_sample()
            if rand < epsilon:
                value_probabilities = (
                    torch.softmax(values_torch_cpu, dim=0, dtype=torch.float32)
                    .squeeze()
                    .detach()
                    .numpy()
                )
                action_idx = np.random.choice(range(len(values)), p=value_probabilities)
            else:
                action_idx = torch.argmax(policy_net_output, dim=0)

            if action_idx == l_actions - 1:
                selected_action = torch.tensor(
                    [[0, 0, TERMINAL_ACTION]], dtype=torch.int8, device=device
                )
            else:
                selected_action = possible_actions[action_idx]
            assert selected_action.shape == (1, 3) or selected_action.shape == (
                3,
            ), selected_action.shape

            selected_value = values_torch_cpu[action_idx]
            assert selected_value.shape == (1, 1) or selected_value.shape == (
                1,
            ), selected_value.shape

            batch_selected_actions[i] = selected_action.squeeze().cpu().numpy()
            batch_selected_values[i] = selected_value.squeeze().cpu().numpy()

    selected_actions = np.stack(batch_selected_actions)
    selected_values = np.stack(batch_selected_values)

    return selected_actions, selected_values


def create_possible_operators(
    possible_action_list: List[Tuple],
    state_size: int,
    stack_depth: int,
    combined_mask: torch.Tensor,
    local_deltas: torch.Tensor,
    device: Union[torch.DeviceObjType, str] = "cpu",
    max_l: int = None,
):
    """
    stacked_sample: shape (L, h, d+1, d+1), or maybe hstacked (h, {d+1}*L, d+1)
    possible_action_list: maximal L-dimensional list with 3-tuples
    max_l: maximum number of possible actions across one batch
        might be larger that {len(possible_action_list) + 1}

    assumes pauli_delta_x = [[0, 1],[1, 0]]
    assumes pauli_delta_y = [[1, 1],[1, 1]]
    assumes pauli_delta_z = [[1, 0],[0, 1]]

    Example:
    possible_action_list = [
        (1,1,1),(1,1,2),(1,1,3),(1,2,1),(1,2,2),(1,2,3),
        (2,1,1),(2,1,2),(2,1,3),(2,2,1),(2,2,2),(2,2,3)
    ]
    """

    if max_l is None:
        max_l = len(possible_action_list) + 1

    # reshape stacked_sample to (h, {d+1}*L, d+1)
    local_deltas.to(device)
    n_possible_actions = len(possible_action_list)

    stacked_combined_mask = torch.tile(
        combined_mask[None, None, :, :], (max_l, stack_depth, 1, 1)
    )

    # collect local syndrome delta matrices
    operators = torch.stack(
        [
            torch.roll(
                torch_pad(
                    local_deltas[operator - 1],
                    (0, state_size - 2, 0, state_size - 2),
                    "constant",
                    0,
                ),  # shift=(x_coord, y_coord)
                shifts=(x_coord, y_coord),
                dims=(0, 1),
            )
            for (x_coord, y_coord, operator) in possible_action_list
        ]
    )

    # repeat operators depth-wise
    operators = torch.tile(operators[:, None, :, :], (1, stack_depth, 1, 1))

    # pad with zero-actions along action-dimension,
    # so that all samples in a batch have the same action-dimensionality
    operators = torch_pad(
        operators, (0, 0, 0, 0, 0, 0, 0, max_l - n_possible_actions), "constant", 0
    )

    operators = operators.to(device)
    stacked_combined_mask = stacked_combined_mask.to(device)

    operators = torch.logical_and(operators, stacked_combined_mask)

    return operators


def format_torch(states, device="cpu", dtype=torch.float32):
    x = states
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype)
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
    elif x.device != device:
        x = x.to(device)

    return x


def determine_possible_actions(
    state: torch.Tensor,
    code_size: int,
    coordinate_shifts: torch.Tensor,
    device: Union[torch.DeviceObjType, str] = "cpu",
):
    """
    Find the action which change the qubits adjacent to active syndromes.

    Return
    ======
    possible_actions: (batch_size, 3) tensor containing action tuples for each sample
    """

    state = format_torch(state, dtype=torch.int8, device=device)
    coordinate_shifts = format_torch(coordinate_shifts, dtype=torch.int8, device=device)
    coordinate_shifts = coordinate_shifts.to(device)

    assert len(state.shape) == 3, state.shape
    # assert coordinate_shifts.device == torch.device(device), f"{coordinate_shifts.device=}, {device=}"

    syndrome_coordinates = torch.nonzero(state)
    syndrome_coordinates = syndrome_coordinates[:, 1:]
    syndrome_coordinates = torch.unique(syndrome_coordinates, dim=0)

    if len(syndrome_coordinates) == 0:
        syndrome_coordinates = torch.randint(0, code_size, size=(2, 2))

    syndrome_coordinates = syndrome_coordinates.to(device)

    possible_qb_coordinates = torch.stack(
        [
            syndrome_coord + shift
            for shift in coordinate_shifts
            for syndrome_coord in syndrome_coordinates
        ]
    )

    possible_qb_coordinates = torch.clip(
        possible_qb_coordinates, min=0, max=code_size - 1
    )
    possible_qb_coordinates = torch.unique(possible_qb_coordinates, dim=0)

    possible_actions = torch.stack(
        [
            torch.tensor([x_coord, y_coord, operator])
            for (x_coord, y_coord) in possible_qb_coordinates
            for operator in (1, 2, 3)
        ]
    )
    return possible_actions
