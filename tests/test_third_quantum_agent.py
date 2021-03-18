import sys
import torch
from src.agent_src.agent3 import QuantumAgent3
sys.path.append("./src")
sys.path.append("./src/surface_rl_decoder")

def test_agent3():

    config = {"size": 5,
    "min_qbit_err": 2,
    "p_error": 0.1,
    "p_msmt": 0.05,
    "stack_depth": 8,
    "nr_actions_per_qubit": 3,
    "epsilon_from": 0.999,
    "epsilon_to": 0.02,
    "epsilon_decay": 0.998,
    "max_actions": 32,
    "input_channels": 1,
    "kernel_size": 3,
    "output_channels": 20,
    "output_channels2": 50,
    "output_channels3": 30,
    "padding_size": 1,
    "lstm_layers": 3
    }
    agent = QuantumAgent3(config)
    tensorX = torch.ones(8,1,6,6)
    tensorZ = torch.zeros(8,1,6,6)
    tensorBoth = tensorX+tensorZ
    agent(tensorBoth)
    print("single input passed")
