"""
This module serves to replicate the surface code environment
multiple times to run steps in parallel.
"""
from copy import deepcopy
from typing import Dict, List, Tuple, Union
import numpy as np
from surface_rl_decoder.surface_code import SurfaceCode


class EnvironmentSet:
    """
    Initialize multiple instances of the surface code environment
    independent of each other.
    """

    def __init__(self, env: SurfaceCode, num_environments):
        self.system_size = env.system_size
        self.stack_depth = env.stack_depth
        self.syndrome_size = env.syndrome_size
        self.num_environments = num_environments
        self._states = np.empty(
            (
                num_environments,
                self.stack_depth,
                self.syndrome_size,
                self.syndrome_size,
            ),
            dtype=np.uint8,
        )

        self.environments = [None] * num_environments
        for i in range(num_environments):
            self.environments[i] = deepcopy(env)

    def reset_terminal_environments(
        self,
        indices: Union[List, np.ndarray],
        p_error: Union[List, None] = None,
        p_msmt: Union[List, None] = None,
    ) -> np.ndarray:
        """
        Given a list of indices denoting environments in a terminal state,
        reset those specific environments.

        Parameters
        ==========
        indices: list of indices, giving the location of terminal environments
        p_error (optional): list of values for the physical error probability
            to set in the different environments
        p_msmt (optional): list of values for the syndrome measurement
            error probability to set in the different environments

        Returns
        =======
        states: the collection of states of all environments in this environment set
        """

        if p_error is None and p_msmt is None:
            for idx in indices:
                self._states[idx] = self.environments[idx].reset()
        else:
            assert p_error is not None, "Both p_error and p_msmt need to be defined"
            assert p_msmt is not None, "Both p_error and p_msmt need to be defined"
            for idx in indices:
                self._states[idx] = self.environments[idx].reset(
                    p_error=p_error[idx], p_msmt=p_msmt[idx]
                )

        return self._states

    def reset_all(
        self, p_error: Union[List, None] = None, p_msmt: Union[List, None] = None
    ) -> np.ndarray:
        """
        Reset all environments in the environment set simultaneously.

        Parameters
        ==========
        p_error (optional): list of values for the physical error probability
            to set in the different environments
        p_msmt (optional): list of values for the syndrome measurement
            error probability to set in the different environments

        Returns
        =======
        states: the collection of states of all environments in this environment set
        """
        if p_error is None and p_msmt is None:
            for i, env in enumerate(self.environments):
                self._states[i] = env.reset()
        else:
            assert p_error is not None, "Both p_error and p_msmt need to be defined"
            assert p_msmt is not None, "Both p_error and p_msmt need to be defined"
            for i, env in enumerate(self.environments):
                self._states[i] = env.reset(p_error=p_error[i], p_msmt=p_msmt[i])
        return self._states

    def step(
        self,
        actions: List,
        discount_intermediate_reward=0.75,
        annealing_intermediate_reward=1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Take a step in each environment instance.

        Parameters
        ==========
        actions: list of action-defining tuples,
            required shape either (# environments, 3) or (# environments, 4)
        discount_intermediate_reward: (optional) discount factor determining how much
            early layers should be discounted when calculating the intermediate reward
        annealing_intermediate_reward: (optional) variable that should decrease over time during
            a training run to decrease the effect of the intermediate reward

        Returns
        =======
        states: list of states for all environments
        rewards: list of rewards obtained in this step for all environments
        terminals: list of terminal flags
        infos: list of dictionaries containing additional information about an environment
        """

        states = np.empty(
            (
                self.num_environments,
                self.stack_depth,
                self.syndrome_size,
                self.syndrome_size,
            ),
            dtype=np.uint8,
        )
        rewards = np.empty(self.num_environments)
        terminals = np.empty(self.num_environments)
        infos = [None] * self.num_environments

        for i, env in enumerate(self.environments):
            next_state, reward, terminal, info = env.step(
                actions[i],
                discount_intermediate_reward=discount_intermediate_reward,
                annealing_intermediate_reward=annealing_intermediate_reward,
            )
            states[i] = next_state
            rewards[i] = reward
            terminals[i] = terminal
            infos[i] = info

        return states, rewards, terminals, infos
