"""
Utility functions for the learner process
"""
from typing import List, Tuple
import numpy as np
import torch
from torch import from_numpy
from distributed.util import action_to_q_value_index


def data_to_batch(
    data: Tuple, device: torch.device
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

    def to_network_input(batch):
        batch_input = np.stack(batch, axis=0)
        tensor = from_numpy(batch_input)
        tensor = tensor.type("torch.Tensor")
        return tensor.to(device)

    # indices:
    # [batch][state, action, reward, next_state, terminal]
    batch = data[0]
    # the following is only meaningful in prioritized experience replay
    memory_weights = data[1]
    if memory_weights is not None:
        memory_weights = torch.tensor(
            memory_weights, dtype=torch.float32, device=device
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


# pylint: disable=too-many-locals, too-many-statements
def perform_q_learning_step(
    policy_net,
    target_net,
    device,
    criterion,
    optimizer,
    data,
    code_size,
    batch_size,
    discount_factor,
    logger=None,
    verbosity=0,
):
    """
    Perform the actual stochastic gradient descent step.
    Make use of a frozen target network to stabilize training.

    Parameters
    ==========
    policy_net: online network to peform the actual training step on
    target_net: offline network with frozen parameters,
        serves as the target Q value term in the Bellman equation.
    device: torch devic
    criterion: loss function
    optimizer: optimizer for training
    data: (Tuple) data received io-learner-queue
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
    ) = data_to_batch(data, device)

    batch_action_indices = torch.tensor(
        [
            action_to_q_value_index(batch_actions[i], code_size)
            for i in range(batch_size)
        ]
    ).view(-1, 1)
    batch_action_indices = batch_action_indices.to(device)

    policy_net.train()
    target_net.eval()

    # compute policy net output
    policy_output = policy_net(batch_state)
    assert policy_output.shape == (
        batch_size,
        3 * code_size * code_size + 1,
    ), policy_output.shape
    policy_output_gathered = policy_output.gather(1, batch_action_indices)

    # compute target network output
    with torch.no_grad():
        target_output = target_net(batch_next_state)
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

    if verbosity > 9:
        logger.info(f"{policy_net.parameters()=}")

    return indices, priorities
