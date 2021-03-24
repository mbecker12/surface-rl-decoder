import json
import torch
import numpy as np
from src.distributed.model_util import extend_model_config, choose_model


def test_conv2d_agent():
    model_name = "conv2d_lstm"
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
