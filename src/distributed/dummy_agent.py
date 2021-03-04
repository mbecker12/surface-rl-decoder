import torch
from torch import nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self, syndrome_size, stack_depth):
        super().__init__()
        self.lin1 = nn.Linear(stack_depth * syndrome_size * syndrome_size, 512, bias=True)
        self.lin2 = nn.Linear(512, 512)
        n_qubits = syndrome_size - 1
        n_actions_total = 3 * n_qubits * n_qubits + 1
        self.output = nn.Linear(512, n_actions_total)

    # pylint: disable=invalid-name
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.output(x)

        return F.sigmoid(x)