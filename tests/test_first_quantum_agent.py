import sys
import torch
from src.agent_src.agent import QuantumAgent1
sys.path.append("./src")
sys.path.append("./src/surface_rl_decoder")


def test_agent2():

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
    "hidden_concat_size":10,
    "lstm_layers": 3,
    "hidden_x": 10,
    "hidden_z": 10,
    "hidden_both": 10
    }
    agent = QuantumAgent1(config)
    tensorX = torch.ones(8,1,6,6)
    tensorZ = torch.zeros(8,1,6,6)
    tensorBoth = tensorX+tensorZ
    agent(tensorBoth)
    print("single input passed")

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
    "hidden_concat_size": 10,
    "lstm_layers": 3,
    "hidden_x": 10,
    "hidden_z": 10,
    "hidden_both": 10
    }
    agent = QuantumAgent1(config)
    tensorX = torch.ones(8,3,6,6)
    tensorZ = torch.zeros(8,3,6,6)
    tensorBoth = tensorX+tensorZ
    agent(tensorBoth)
    print("batched input passed")

