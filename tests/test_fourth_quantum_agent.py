import sys
import torch
from src.agent_src import Conv3dGeneralAgent
sys.path.append("./src")
sys.path.append("./src/surface_rl_decoder")

def test_agent3():

    config = {"code_size": 5,
    "device": cpu,
    "split_input_toggle": 1,
    "min_qbit_err": 2,
    "p_error": 0.1,
    "p_msmt": 0.05,
    "stack_depth": 8,
    "num_actions_per_qubit": 3,
    "epsilon_from": 0.999,
    "epsilon_to": 0.02,
    "epsilon_decay": 0.998,
    "max_actions": 32,
    "input_channels": 1,
    "kernel_size": 3,
    "output_channels": 20,
    "output_channels2": 50,
    "output_channels3": 30,
    "output_channels4": 3,
    "padding_size": 1
    }
    agent = Conv3dGeneralAgent(config)
    tensorX = torch.ones(8,3,6,6)
    tensorZ = torch.zeros(8,3,6,6)
    tensorBoth = tensorX+tensorZ
    output = agent(tensorBoth)
    assert output.shape == (3,3*5**2+1)
    print("single input passed")
