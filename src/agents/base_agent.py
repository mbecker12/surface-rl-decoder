from abc import ABC, abstractmethod
import numpy as np
# pylint: disable=not-callable
import torch
from torch import nn

from distributed.util import q_value_index_to_action


class BaseAgent(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.code_size = None

    def _format(self, states, device=None):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    @abstractmethod
    def forward(self, x):
        # To be overwritten
        return x

    def np_pass(self, states):
        logits, values = self.forward(states)
        np_values = values.detach().cpu().numpy()
        np_logits = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        logpas = dist.log_prob(actions)
        np_logpas = logpas.detach().cpu().numpy()
        is_exploratory = np_actions != np.argmax(np_logits, axis=1)
        # TODO: convert action to 3-tuple
        # TODO: define code_size
        np_action_tuples = np.array([
            q_value_index_to_action(ac, self.code_size) for _, ac in enumerate(np_actions)
        ])
        return np_action_tuples, np_logpas, is_exploratory, np_values

    def select_action_ppo(self, states):
        logits, _ = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        # TODO: think about using q_value_index_to_action() here
        return action.detach().cpu().item()

    def get_predictions_ppo(self, states, actions):
        # TODO: think about using q_value_index_to_action() or its inverse here
        states, actions = self._format(states), self._format(actions)
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        return logpas, entropies, values

    def select_greedy_action_ppo(self, states):
        logits, _ = self.forward(states)
        # TODO: think about using q_value_index_to_action() here
        return np.argmax(logits.detach().squeeze().cpu().numpy())
