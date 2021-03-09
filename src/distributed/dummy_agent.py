"""
Stand-in dummy agent, just to make sure that data is being processed
by a torch nn.Module subclass
"""
from torch import nn
import torch
import torch.nn.functional as F


class DummyModel(nn.Module):
    """
    Just a dummy model to make sure that the data can be processed and
    that the different subprocesses involved run properly.
    """
    def __init__(self, syndrome_size, stack_depth, num_actions_per_qubit=3):
        super().__init__()
        self.syndrome_size = syndrome_size
        self.stack_depth = stack_depth
        n_qubits = syndrome_size - 1

        self.lin1 = nn.Linear(
            stack_depth * syndrome_size * syndrome_size, 512, bias=True
        )
        self.lin2 = nn.Linear(512, 512)
        n_actions_total = num_actions_per_qubit * n_qubits * n_qubits + 1
        self.output = nn.Linear(512, n_actions_total)

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor):
        """
        Bog-standard forward method for torch neural networks
        """
        # flatten the syndrome stack as a dummy operation to make the shapes fit
        x = x.view((-1, self.stack_depth * self.syndrome_size * self.syndrome_size))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.output(x)

        return x
