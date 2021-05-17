from abc import ABC, abstractmethod, abstractproperty
import numpy as np

# pylint: disable=not-callable
import torch
from torch import nn

from distributed.util import q_value_index_to_action


class BaseAgent(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.size = None
        self.device = None

    def _format(self, states):
        assert self.device is not None
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        elif x.device != self.device:
            x.to(self.device)

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
        np_action_tuples = np.array(
            [
                q_value_index_to_action(ac, self.size)
                for _, ac in enumerate(np_actions)
            ]
        )
        return np_action_tuples, np_logpas, is_exploratory, np_values

    def select_action_ppo(self, states, return_logits=False, return_values=False):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        detached_actions = action.detach().cpu().squeeze()
        if return_logits:
            if return_values:
                return detached_actions, logits, values
            return detached_actions, logits

        if return_values:
            return detached_actions, None, values
        return detached_actions

    def get_predictions_ppo(self, states, actions):
        states = self._format(states)
        actions = self._format(actions)
        actions = actions.squeeze()
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        # dist.to(self.device)
        # actions.to(self.device)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        return logpas, entropies, values

    def select_greedy_action_ppo(self, states, return_logits=False, return_values=False):
        logits, values = self.forward(states)
        action_index = np.argmax(logits.detach().squeeze().cpu().numpy(), axis=1)
        if return_logits:
            if return_values:
                return action_index, logits, values
            return action_index, logits

        if return_values:
            return action_index, None, values
        return action_index