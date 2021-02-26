from surface_rl_decoder.surface_code import SurfaceCode
from copy import deepcopy
import numpy as np
from typing import Dict, List, Tuple, Union


class EnvironmentSet:
    def __init__(self, env: SurfaceCode, num_environments):
        self.system_size = env.system_size
        self.stack_depth = env.stack_depth
        self.syndrome_size = env.syndrome_size
        self.num_environments = num_environments
        self.states = np.empty(
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

        if p_error is None and p_msmt is None:
            for idx in indices:
                self.states[idx] = self.environments[idx].reset()
        else:
            assert p_error is not None, "Both p_error and p_msmt need to be defined"
            assert p_msmt is not None, "Both p_error and p_msmt need to be defined"
            for idx in indices:
                self.states[idx] = self.environments[idx].reset(
                    p_error=p_error[idx], p_msmt=p_msmt[idx]
                )

        return self.states

    def reset_all(
        self, p_error: Union[List, None] = None, p_msmt: Union[List, None] = None
    ) -> np.ndarray:
        if p_error is None and p_msmt is None:
            for i, env in enumerate(self.environments):
                self.states[i] = env.reset()
        else:
            assert p_error is not None, "Both p_error and p_msmt need to be defined"
            assert p_msmt is not None, "Both p_error and p_msmt need to be defined"
            for i, env in enumerate(self.environments):
                self.states[i] = env.reset(p_error=p_error[i], p_msmt=p_msmt[i])
        return self.states

    def step(self, actions: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
            next_state, reward, terminal, info = env.step(actions[i])
            states[i] = next_state
            rewards[i] = reward
            terminals[i] = terminal
            infos[i] = info

        return states, rewards, terminals, infos
