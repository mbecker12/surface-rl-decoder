import json
import torch
import numpy as np
from src.distributed.model_util import extend_model_config, choose_model


def test_conv2d_agent():
    model_name = "conv2d"
    with open("src/config/model_spec/conv_agents.json") as model_conf_file:
        model_config = json.load(model_conf_file)[model_name]

    syndrome_size = 6
    code_size = syndrome_size - 1
    stack_depth = 8
    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model = choose_model(model_name, model_config)

    batch_size = 16
    state = np.random.randint(
        0, 2, size=(batch_size, stack_depth, syndrome_size, syndrome_size)
    )
    state = torch.tensor(state, dtype=torch.float32)
    output = model(state)
    assert output.shape == (batch_size, 3 * code_size * code_size + 1)


def test_conv2d_agent_w_opposite_split_toggle():
    model_name = "conv2d"
    with open("src/config/model_spec/conv_agents.json") as model_conf_file:
        model_config = json.load(model_conf_file)[model_name]

    assert model_config["split_input_toggle"] in (0, 1)
    split_input_toggle_before = model_config["split_input_toggle"]
    model_config["split_input_toggle"] = 1 - model_config["split_input_toggle"]
    split_input_toggle_after = model_config["split_input_toggle"]
    assert split_input_toggle_before != split_input_toggle_after
    assert split_input_toggle_before + split_input_toggle_after == 1

    syndrome_size = 6
    code_size = syndrome_size - 1
    stack_depth = 8
    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model = choose_model(model_name, model_config)

    batch_size = 16
    state = np.random.randint(
        0, 2, size=(batch_size, stack_depth, syndrome_size, syndrome_size)
    )
    state = torch.tensor(state, dtype=torch.float32)
    output = model(state)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    assert output.shape == (batch_size, 3 * code_size * code_size + 1)


def test_conv2d_agent_lstm():
    model_name = "conv2d"
    with open("src/config/model_spec/conv_agents.json") as model_conf_file:
        model_config = json.load(model_conf_file)[model_name]

    syndrome_size = 6
    code_size = syndrome_size - 1
    stack_depth = 8
    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model_config["use_lstm"] = 1
    model_config["use_rnn"] = 0
    model_config["use_gtrxl"] = 0
    model = choose_model(model_name, model_config)

    batch_size = 16
    state = np.random.randint(
        0, 2, size=(batch_size, stack_depth, syndrome_size, syndrome_size)
    )
    state = torch.tensor(state, dtype=torch.float32)
    output = model(state)
    assert output.shape == (batch_size, 3 * code_size * code_size + 1)


def test_conv2d_agent_gtrxl():
    model_name = "conv2d"
    with open("src/config/model_spec/conv_agents.json") as model_conf_file:
        model_config = json.load(model_conf_file)[model_name]

    syndrome_size = 6
    code_size = syndrome_size - 1
    stack_depth = 8
    model_config = extend_model_config(model_config, syndrome_size, stack_depth)
    model_config["use_lstm"] = 0
    model_config["use_rnn"] = 0
    model_config["use_gtrxl"] = 1
    model = choose_model(model_name, model_config)

    batch_size = 16
    state = np.random.randint(
        0, 2, size=(batch_size, stack_depth, syndrome_size, syndrome_size)
    )
    state = torch.tensor(state, dtype=torch.float32)
    output = model(state)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    assert output.shape == (batch_size, 3 * code_size * code_size + 1)
