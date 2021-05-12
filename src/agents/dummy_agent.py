"""
Stand-in dummy agent, just to make sure that data is being processed
by a torch nn.Module subclass
"""
from torch import nn
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent


class DummyModel(BaseAgent):
    """
    Just a dummy model to make sure that the data can be processed and
    that the different subprocesses involved run properly.
    """

    def __init__(self, config):
        super().__init__()
        self.syndrome_size = config["syndrome_size"]
        self.code_size = self.syndrome_size - 1
        self.stack_depth = config["stack_depth"]
        self.size = config["code_size"]
        # self.code_size = self.syndrome_size - 1
        num_actions_per_qubit = config["num_actions_per_qubit"]
        layer1_size = config["layer1_size"]
        layer2_size = config["layer2_size"]
        self.device = config["device"]
        self.rl_type = config.get("rl_type", "q")

        self.lin1 = nn.Linear(
            self.stack_depth * self.syndrome_size * self.syndrome_size,
            layer1_size,
            bias=True,
        )
        self.lin2 = nn.Linear(layer1_size, layer2_size)
        n_actions_total = num_actions_per_qubit * self.code_size * self.code_size + 1
        self.output = nn.Linear(layer1_size, n_actions_total)
        if "ppo" in self.rl_type.lower():
            self.value_output = nn.Linear(layer1_size, 1)

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor):
        """
        Bog-standard forward method for torch neural networks
        """
        x = self._format(x, device=self.device)
        # flatten the syndrome stack as a dummy operation to make the shapes fit
        x0 = x.view((-1, self.stack_depth * self.syndrome_size * self.syndrome_size))
        x1 = F.relu(self.lin1(x0))
        x2 = F.relu(self.lin2(x1))

        if "ppo" in self.rl_type.lower():
            v0 = self.value_output(x2) * 100
            x_out = self.output(x2)
            return x_out, v0
        else:
            x_out = self.output(x2)
            return x_out
