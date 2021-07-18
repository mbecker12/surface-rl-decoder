import json
import numpy as np
import pytest
import torch
from agents.conv_2d_agent_vnet import Conv2dAgentValueNet

from distributed.util import (
    COORDINATE_SHIFTS,
    LOCAL_DELTAS,
    determine_possible_actions,
    format_torch,
    get_successor_states,
    select_actions_value_network,
)
from distributed.model_util import extend_model_config
from surface_rl_decoder.syndrome_masks import get_plaquette_mask, get_vertex_mask


batch_size = 4
stack_depth = 1
code_size = 5
syndrome_size = code_size + 1


def test_select_action():
    state_batch = torch.zeros(
        (batch_size, stack_depth, syndrome_size, syndrome_size), dtype=torch.float32
    )

    rand = torch.rand(state_batch.shape, dtype=torch.float32)
    state_batch = (rand < 0.3).to(dtype=torch.float32)
    # state_batch = 2 * state_batch - 1
    print(f"{state_batch.dtype=}\n{state_batch=}")

    model_config_file_path = "src/config/model_spec/conv_agents_slim.json"
    with open(model_config_file_path, "r") as model_config_file:
        model_config = json.load(model_config_file)["conv2d"]

    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model_config["device"] = "cpu"
    model_config["input_channels"] = 1
    model_config["rl_type"] = "v"

    model = Conv2dAgentValueNet(model_config)
    device = torch.device("cpu")

    vertex_mask = get_vertex_mask(code_size)
    plaquette_mask = get_plaquette_mask(code_size)
    combined_mask = np.logical_or(vertex_mask, plaquette_mask)
    combined_mask = format_torch(combined_mask, device=device, dtype=torch.int8)

    (
        selected_actions,
        selected_values,
        optimal_actions,
        optimal_values,
    ) = select_actions_value_network(
        state_batch,
        model,
        code_size,
        stack_depth,
        combined_mask,
        COORDINATE_SHIFTS,
        LOCAL_DELTAS,
        device,
        epsilon=0.5,
    )

    assert selected_actions.shape == (batch_size, 3)
    assert selected_values.shape == (batch_size, 1) or selected_values.shape == (
        batch_size,
    )

    assert optimal_actions.shape == (batch_size, 3)
    assert optimal_values.shape == (batch_size, 1) or optimal_values.shape == (
        batch_size,
    )

    assert not np.any(selected_values > optimal_values)


def test_determine_possible_actions():
    state_batch = torch.zeros(
        (batch_size, stack_depth, syndrome_size, syndrome_size), dtype=torch.float32
    )

    state = state_batch[0]
    assert torch.all(state == 0)
    possible_actions = determine_possible_actions(
        state,
        code_size,
        coordinate_shifts=COORDINATE_SHIFTS,
        device=torch.device("cpu"),
    )

    print(f"{possible_actions.shape=}")
    print(f"{possible_actions=}")

    assert possible_actions.shape[1] == 3


def test_get_successor_states():
    state_batch = torch.zeros(
        (batch_size, stack_depth, syndrome_size, syndrome_size), dtype=torch.float32
    )

    state = state_batch[0]
    device = torch.device("cpu")

    possible_actions = determine_possible_actions(
        state, code_size, coordinate_shifts=COORDINATE_SHIFTS, device=device
    )
    l_actions = len(possible_actions) + 1

    assert l_actions == 2

    vertex_mask = get_vertex_mask(code_size)
    plaquette_mask = get_plaquette_mask(code_size)
    combined_mask = np.logical_or(vertex_mask, plaquette_mask)
    combined_mask = format_torch(combined_mask, device=device, dtype=torch.int8)

    new_states = get_successor_states(
        state,
        possible_actions,
        l_actions,
        code_size,
        stack_depth,
        combined_mask,
        LOCAL_DELTAS,
        device,
    )

    model_config_file_path = "src/config/model_spec/conv_agents_slim.json"
    with open(model_config_file_path, "r") as model_config_file:
        model_config = json.load(model_config_file)["conv2d"]

    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model_config["device"] = "cpu"
    model_config["input_channels"] = 1
    model_config["rl_type"] = "v"

    model = Conv2dAgentValueNet(model_config)

    policy_net_output = model(new_states)
    values_torch_cpu = policy_net_output.detach().cpu()
    values = np.array(values_torch_cpu)
    print(f"{new_states.shape=}")
    print(f"{possible_actions.dtype=}")
    for some_idx, ns in enumerate(new_states):
        try:
            print(f"{values[some_idx]} {possible_actions[some_idx]} -> {ns}")
        except:
            print(f"{values[some_idx]} ... -> {ns}")
