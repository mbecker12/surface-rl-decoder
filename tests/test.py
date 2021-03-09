import torch
from src.surface_rl_decoder.agent_src.agent import QuantumAgent1


def test_agent2():
    agent = QuantumAgent1()
    tensorX = torch.ones(8,1,6,6)
    tensorZ = torch.zeros(8,1,6,6)
    tensorBoth = tensorX+tensorZ
    agent(tensorX, tensorZ, tensorBoth)
    print("single input passed")

def test_agent3():
    agent = QuantumAgent1()
    tensorX = torch.ones(8,3,6,6)
    tensorZ = torch.zeros(8,3,6,6)
    tensorBoth = tensorX+tensorZ
    agent(tensorX, tensorZ, tensorBoth, batch_size = 3)
    print("single input passed")

