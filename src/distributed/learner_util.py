"""
Utility functions for the learner process
"""
from typing import List, Tuple
import numpy as np
import torch

# pylint: disable=no-name-in-module
from torch import from_numpy
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

        memory_weights = (
            torch.tensor(memory_weights, dtype=torch.float32, device=device)
            .clone()
            .detach()
        )
        memory_weights = memory_weights.view(-1, 1)

    indices = data[2]

    list_state, list_action, list_reward, list_next_state, list_terminal = zip(*batch)

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

    target_q_values = expected_q_values * batch_reward
    target_q_values = target_q_values.view(-1, 1)
    target_q_values = target_q_values.clamp(-100, 100)

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
    tensorboard,
    list_of_p_errors,
    episode_results,
    step_results,
    p_error_results,
    evaluation_step,
    current_time_ms,
):
    """
    Utility function to send the evaluation data to tensorboard.
    """
    for i, p_err in enumerate(list_of_p_errors):
        tensorboard.add_scalars(
            f"network/episode, p_error {p_err}",
            episode_results[i],
            evaluation_step,
            walltime=current_time_ms,
        )

        tensorboard.add_scalars(
            f"network/step, p_error {p_err}",
            step_results[i],
            evaluation_step,
            walltime=current_time_ms,
        )

        tensorboard.add_scalars(
            f"network/p_err, p_error {p_err}",
            p_error_results[i],
            evaluation_step,
            walltime=current_time_ms,
        )
