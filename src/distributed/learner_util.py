"""
Utility functions for the learner process
"""
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# pylint: disable=no-name-in-module
from torch import from_numpy
from torch.utils.tensorboard import SummaryWriter
from distributed.util import action_to_q_value_index


def data_to_batch(
    data: Tuple, device: torch.device, batch_size: int
) -> Tuple[List, List, List, List, List, List, List]:
    """
    Transform the data received from the io-learner-queue to data forms
    that can be processed by the agent.
    The data will be stacked in batches, transformed to torch tensors
    and moved to the device.

    Parameters
    ==========
    data: (Tuple) consisting of
        (
            transition batches,
            memory weights [for prioritized experience replay],
            indices [for prioritized experience replay]
        )
    device: torch device

    Returns
    =======
    batch_state: tensor of shape (batch_size, stack_depth, state_size, state_size)
        batch of starting syndrome states
    batch_actions: tensor of (batch_size, 3), containing (x-coordinate, y-coordinate, operator)
        batch of actions to perform on the whole stack
    batch_reward: tensor of size (batch_size) [or (batch_size, 1)], dtype float
        batch of rewards for the chosen actions
    batch_next_state: tensor of shape (batch_size, stack_depth, state_size, state_size)
        batch of resulting syndrome states
    batch_terminal: tensor of size (batch_size) [or (batch_size, 1)], dtype bool,
        batch of terminal flags for each transition
    memory_weights: values of memory weights to alter the loss value for backpropagation
    indices: indices of transitions in memory replay
    """
    # because we can indeed call torch.tensor()  -_- ...
    # pylint: disable=not-callable
    def to_network_input(batch):
        batch_input = np.stack(batch, axis=0)
        tensor = from_numpy(batch_input)
        tensor = tensor.type("torch.Tensor")
        return tensor.to(device)

    # indices:
    # [batch][state, action, reward, next_state, terminal]
    batch = data[0]
    assert batch is not None and len(batch) == batch_size

    # the following is only meaningful in prioritized experience replay
    memory_weights = data[1]
    if memory_weights is not None:
        assert len(memory_weights) == batch_size, len(memory_weights)
        if isinstance(memory_weights, torch.Tensor):
            memory_weights = memory_weights.clone().detach().float().to(device)
        else:
            memory_weights = torch.tensor(
                memory_weights, dtype=torch.float32, device=device
            )
        memory_weights = memory_weights.view(-1, 1)

    indices = data[2]

    # pylint: disable=bare-except
    try:
        list_state, list_action, list_reward, list_next_state, list_terminal = zip(
            *batch
        )
    except:
        print(f"{list(zip(*batch))=}")

    batch_state = to_network_input(list_state)
    batch_next_state = to_network_input(list_next_state)

    batch_action = torch.tensor(list_action, dtype=torch.int64, device=device)
    batch_terminal = from_numpy(np.array(list_terminal)).to(device)
    batch_reward = from_numpy(np.array(list_reward)).type("torch.Tensor").to(device)

    return (
        batch_state,
        batch_action,
        batch_reward,
        batch_next_state,
        batch_terminal,
        memory_weights,
        indices,
    )


# pylint: disable=too-many-locals, too-many-statements, too-many-arguments
def perform_q_learning_step(
    policy_network,
    target_network,
    device,
    criterion,
    optimizer,
    input_data,
    code_size,
    batch_size,
    discount_factor,
    policy_model_max_grad_norm=100,
):
    """
    Perform the actual stochastic gradient descent step.
    Make use of a frozen target network to stabilize training.

    Parameters
    ==========
    policy_net: online network to peform the actual training step on
    target_net: offline network with frozen parameters,
        serves as the target Q value term in the Bellman equation.
    device: torch device
    criterion: loss function
    optimizer: optimizer for training
    input_data: (Tuple) data received io-learner-queue
    code_size: code distance, number of qubits in one row/column
    batch_size: number of different states in a batch
    discount_factor: γ-factor in reinforcement learning

    Returns
    =======
    indices: indices for (prioritized) memory replay objects
    priorities: priorities for (prioritized) memory replay objects
    """
    (
        batch_state,
        batch_actions,
        batch_reward,
        batch_next_state,
        batch_terminal,
        weights,
        indices,
    ) = data_to_batch(input_data, device, batch_size)

    # pylint: disable=not-callable
    batch_action_indices = torch.tensor(
        [
            action_to_q_value_index(batch_actions[i], code_size)
            for i in range(batch_size)
        ]
    ).view(-1, 1)
    batch_action_indices = batch_action_indices.to(device)

    policy_network.train()
    target_network.eval()

    # compute policy net output
    policy_output = policy_network(batch_state)
    assert policy_output.shape == (
        batch_size,
        3 * code_size * code_size + 1,
    ), policy_output.shape
    policy_output_gathered = policy_output.gather(1, batch_action_indices)

    # compute target network output
    with torch.no_grad():
        target_output = target_network(batch_next_state)
        target_output = target_output.max(1)[0].detach()

    # compute loss and update replay memory
    expected_q_values = (
        target_output * (~batch_terminal).type(torch.float32) * discount_factor
    )

    target_q_values = expected_q_values + batch_reward
    target_q_values = target_q_values.view(-1, 1)
    target_q_values = target_q_values.clamp(-200, 200)

    loss = criterion(target_q_values, policy_output_gathered)

    optimizer.zero_grad()

    # only used for prioritized experience replay
    if weights is not None:
        loss = weights * loss

    # Compute priorities
    priorities = np.absolute(loss.cpu().detach().numpy())

    loss = loss.mean()

    # backpropagate
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        policy_network.parameters(), policy_model_max_grad_norm
    )
    optimizer.step()

    return indices, priorities


def transform_list_dict(mapping):
    """
    Transform a dictionary of lists to a
    list of dictionaries.

    E.g.
    {
        'result1': array([10, 11, 12]),
        'result2': array([20, 21, 22]),
        'result3': array([30, 31, 32])
    }

    will be transformed to
    [
        {
            'result1': 10,
            'result2': 20,
            'result3': 30
        },
        {
            'result1': 11,
            'result2': 21,
            'result3': 31
        },
        {
            'result1': 12,
            'result2': 22,
            'result3': 32
        }
    ]
    """
    return [dict(zip(mapping, t)) for t in zip(*mapping.values())]


def safe_append_in_dict(dictionary, key, val):
    """
    Check if a list already exists for a given key. In that case append
    val to that list, otherwise create a list with val as the first element

    Parameters
    ==========
    dictionary: target dictionary to be filled with new value
    key: key for the dictionary
    val: value to append to the list whch should exist or be created at the key

    Returns
    =======
    dictionary: modified version of the input dictionary,
        with added value to the target key
    """
    existing_result = dictionary.get(key)
    if existing_result is None:
        dictionary[key] = [val]
    else:
        dictionary[key].append(val)
    return dictionary


def log_evaluation_data(
    tensorboard: SummaryWriter,
    all_results: Dict,
    list_of_p_errors: List,
    evaluation_step: int,
    current_time_tb,
):
    """
    Utility function to send the evaluation data to tensorboard.
    """
    for i, p_err in enumerate(list_of_p_errors):
        for result_key, result_values in all_results.items():
            tensorboard.add_scalars(
                f"network/{result_key}, p_error {p_err}",
                result_values[i],
                evaluation_step,
                walltime=current_time_tb,
            )

# Value Network Functions
def prepare_successor_states(qubits):

    pass


from iniparser import Config
c = Config()
_config = c.scan(".", True).read()
config = c.config_rendered

env_config = config.get("config").get("env")
learner_config = config.get("config").get("learner")
device = learner_config.get("device")
d = int(env_config.get("size"))

# pylint: disable=not-callable
LOCAL_X_DELTA = torch.tensor([[0, 1],[1, 0]], dtype=torch.int8, device=device)
LOCAL_Y_DELTA = torch.tensor([[1, 1],[1, 1]], dtype=torch.int8, device=device)
LOCAL_Z_DELTA = torch.tensor([[1, 0],[0, 1]], dtype=torch.int8, device=device)

LOCAL_DELTAS = torch.stack(
    [LOCAL_X_DELTA, LOCAL_Y_DELTA, LOCAL_Z_DELTA]
)

COORDINATE_SHIFT_1 = [-1, -1]
COORDINATE_SHIFT_2 = [-1, 0]
COORDINATE_SHIFT_3 = [0, -1]
COORDINATE_SHIFT_4 = [0, 0]
COORDINATE_SHIFTS = torch.tensor(
    [COORDINATE_SHIFT_1, COORDINATE_SHIFT_2, COORDINATE_SHIFT_3, COORDINATE_SHIFT_4],
    dtype=torch.int8, device=device
)

from torch.nn.functional import pad as torch_pad
def create_possible_operators(
    possible_action_list: List[Tuple],
    max_l: int,
    state_size: int,
    stack_depth: int,
    combined_mask: torch.Tensor
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

    # reshape stacked_sample to (h, {d+1}*L, d+1)

    n_possible_actions = len(possible_action_list)

    stacked_combined_mask = torch.tile(
        combined_mask[None, None, :, :],
        (max_l, stack_depth, 1, 1)
    )

    # TODO: stack operators properly

    # collect local syndrome delta matrices
    operators = torch.stack(
        [
            torch.roll(
                torch_pad(
                    LOCAL_DELTAS[operator-1], (0, state_size-2, 0, state_size-2), "constant", 0
                ), #shift=(x_coord, y_coord)
                shifts=(x_coord, y_coord),
                dims=(0, 1)
            ) for (x_coord, y_coord, operator) in possible_action_list
        ]
    )

    # repeat operators depth-wise
    operators = torch.tile(
        operators[:, None, :, :],
        (1, stack_depth, 1, 1)
    )

    # pad with zero-actions along action-dimension,
    # so that all samples in a batch have the same action-dimensionality
    operators = torch_pad(
        operators,
        (0, 0, 0, 0, 0, 0, 0, max_l - n_possible_actions),
        "constant",
        0
    )

    operators = torch.logical_and(operators, stacked_combined_mask)
    # print(f"{operators}")

    return operators

def format_torch(states, device="cpu", dtype=torch.float32):
    x = states
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype)
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
    elif x.device != device:
        x.to(device)

    return x

def determine_possible_actions(
    state: torch.Tensor,
    code_size: int
):
    """
    Find the action which change the qubits adjacent to active syndromes.
    """
    state = format_torch(state, dtype=torch.int8, device="cpu")
    syndrome_coordinates = torch.nonzero(state)
    syndrome_coordinates = syndrome_coordinates[:, 1:]
    syndrome_coordinates = torch.unique(syndrome_coordinates, dim=0)
    print(f"{syndrome_coordinates=}")

    possible_qb_coordinates = torch.stack(
        [
            syndrome_coord + shift
            for shift in COORDINATE_SHIFTS
            for syndrome_coord in syndrome_coordinates
        ]
    )

    print(f"{possible_qb_coordinates=}")
    print(f"{possible_qb_coordinates.dtype=}")
    print(f"{possible_qb_coordinates.device=}")

    possible_qb_coordinates = torch.clip(possible_qb_coordinates, min=0, max=code_size-1)
    possible_qb_coordinates = torch.unique(possible_qb_coordinates, dim=0)

    possible_actions = torch.stack(
        [
            torch.tensor([x_coord, y_coord, operator])
            for (x_coord, y_coord) in possible_qb_coordinates
            for operator in (1, 2, 3)
        ]
    )

    print(f"{possible_actions=}")
    return possible_actions


if __name__ == "__main__":
    from surface_rl_decoder.syndrome_masks import get_plaquette_mask, get_vertex_mask
    stack_depth = 4
    state_size = 6
    code_size = state_size - 1
    plaquette_mask = format_torch(get_plaquette_mask(code_size))
    vertex_mask = format_torch(get_vertex_mask(code_size))
    combined_mask = torch.logical_or(plaquette_mask, vertex_mask)
    
    possible_actions = np.array([
        (1,1,1),(1,1,2),(1,1,3),(1,2,1),(1,2,2),(1,2,3),
        (2,1,1),(2,1,2),(2,1,3),(2,2,1),(2,2,2),(2,2,3),
        (0,0,1),(0,0,2),(0,0,3)
    ])
    l_actions = len(possible_actions) + 1
    # print(f"{l_actions=}")
    
    sample = np.zeros((stack_depth, state_size, state_size), dtype=int)
    sample[:, 2, 2] = 1
    sample[:, 3, 3] = 1

    possible_actions = determine_possible_actions(sample, code_size)
    l_actions = len(possible_actions) + 1

    sample = sample[None, :, :, :]
    sample = format_torch(sample, dtype=torch.int8, device="cpu")

    stacked_sample = torch.tile(
        sample,
        (l_actions, 1, 1, 1),
    )
    
    operators = create_possible_operators(
        possible_actions,
        l_actions,
        state_size,
        stack_depth,
        combined_mask
    )

    new_states = torch.tensor(
        torch.logical_xor(stacked_sample, operators),
        dtype=torch.int8, device="cpu"
    )
    
    print(f"{new_states.shape=}")
    print(f"{new_states.dtype=}")

    # Now new_states is already in a shape where the 0th dimension
    # can be interpreted as the batch size.
    # This whole stack can now be fed to a neural network.