import json
import numpy as np
import torch
from src.surface_rl_decoder.syndrome_masks import get_plaquette_mask, get_vertex_mask
from src.distributed.model_util import choose_model, extend_model_config
from src.distributed.util import (
    COORDINATE_SHIFTS,
    LOCAL_DELTAS,
    create_possible_operators,
    determine_possible_actions,
    format_torch,
    select_actions_value_network,
)


def test_select_action():
    batch_size = 4
    stack_depth = 5
    code_size = 5
    state_size = code_size + 1
    all_states = torch.zeros((batch_size, stack_depth, state_size, state_size))

    # leave index 0 as empty state
    # index 1
    all_states[1, :, 1, 2] = 1
    all_states[1, :, 2, 3] = 1

    # index 2
    all_states[2, :, 3, 3] = 1
    all_states[2, :, 4, 3] = 1

    # index 3
    all_states[3, :, 3, 3] = 1
    all_states[3, :, 4, 2] = 1
    all_states[3, :, 1, 1] = 1
    all_states[3, :, 2, 2] = 1

    plaquettes = get_plaquette_mask(code_size)
    vertices = get_vertex_mask(code_size)
    combined_mask = np.logical_or(plaquettes, vertices, dtype=np.int8)
    combined_mask = format_torch(combined_mask, device="cpu", dtype=torch.int8)
    print(f"cp 1: {combined_mask=}")
    model_name = "conv3d"
    model_config_path_3d = "src/config/model_spec/conv_agents_slim.json"

    with open(model_config_path_3d, "r") as jsonfile:
        model_config_3d = json.load(jsonfile)[model_name]

    code_size, stack_depth = 5, 5
    syndrome_size = code_size + 1
    model_config_3d = extend_model_config(model_config_3d, syndrome_size, stack_depth)
    model_config_3d["network_size"] = "slim"
    model_config_3d["rl_type"] = "v"

    model = choose_model(model_name, model_config_3d, transfer_learning=0)

    selected_actions, selected_values = select_actions_value_network(
        all_states,
        model,
        code_size,
        stack_depth,
        combined_mask,
        COORDINATE_SHIFTS,
        LOCAL_DELTAS,
        device="cpu",
    )
    assert selected_actions.shape == (batch_size, 3), selected_actions.shape
    assert selected_values.shape == (batch_size, 1), selected_values.shape
    print(f"cp 2: {combined_mask=}")

    for state in all_states:
        possible_actions = determine_possible_actions(
            state, code_size, COORDINATE_SHIFTS, device="cpu"
        )
        l_actions = len(possible_actions) + 1

        print(f"cp 3: {combined_mask=}")

        operators = create_possible_operators(
            possible_actions,
            state_size,
            stack_depth,
            combined_mask,
            LOCAL_DELTAS,
            device="cpu",
        )

        stacked_sample = torch.tile(
            state,
            (l_actions, 1, 1, 1),
        )
        new_states = torch.tensor(
            torch.logical_xor(stacked_sample, operators), dtype=torch.int8, device="cpu"
        )
