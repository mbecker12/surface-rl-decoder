import gym
import numpy as np
import matplotlib.pyplot as plt
from iniparser import Config
import time, sys
from syndrome_masks import vertex_mask, plaquette_mask


class SurfaceCode(gym.Env):
    """
    Description:
        Defines the rotated surface code for qubit encoding.

        New in this iteration of the project:
            The state now consists of multiple layers of syndrome slices.
            These layers correspond to different time steps, with the oldest time step at index 0.
            The resulting syndrome stack has a fixed height h (i.e. a fixed number of time steps that we keep track of).
            There might be the possibility that the actual error chain is shorter than the stack height. In this case,
            say with k faulty time slices, where k < h, only the k latest slices possess non-zero entries.
            Slices 0 up to h-k are all zero in this case.

            An action will act on one qubit throughout the whole stack.
            The goal of the decoder should be to get rid of all qubit erros by the latest slice (i.e. at index h-1).
            The decoder should learn to detect measurement errors and in such a case rightfully ignore those.
            Hence, it could happen that the latest slice contains errors of the measurement kind after the decoder
            has finished its job; this would be okay.


            # TODO need to keep track of measurement errors separately, so that we know at prediction time
            # which errors are real.

    Actions:
        An action can be a Pauli X, Y, Z, or Identity on any qubit on the surface.

    Reward:

    Episode Termination:
        #TODO Either if the agent decides that it is terminated or if the last remaining surface is error free.

    """

    def __init__(self):
        """
        Initialize Surface Code environment.
        Loads configuration via config-env-parsers, therefore we
        need either config.ini file or constants saved as env variables.
        """

        c = Config()
        _config = c.scan(".", True).read()
        self.config = c.config_rendered.get("config")

        env_config = self.config.get("env")

        self.system_size = int(env_config.get("size"))
        self.min_qbit_errors = int(env_config.get("min_qbit_err"))
        self.p_error = float(env_config.get("p_error"))
        self.p_msmt = float(env_config.get("p_msmt"))
        self.stack_depth = int(env_config.get("stack_depth"))

        # Sweke definition
        self.num_actions = 3 * self.system_size ** 2 + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.completed_actions = np.zeros(self.num_actions, np.uint8)

        self.volume_depth = 3  # what is this? TODO
        self.n_action_layers = (
            3  # what is this? In the case with Y errors, this is 3 TODO
        )

        # observation space should correspond to the shape
        # of vertex- and plaquette-representation
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                self.stack_depth,
                2 * self.system_size + 1,
                2 * self.system_size + 1,
            ),
            dtype=np.uint8,
        )

        self.vertex_mask = vertex_mask
        self.plaquette_mask = plaquette_mask
        assert vertex_mask.shape == (self.system_size + 1, self.system_size + 1)
        assert plaquette_mask.shape == (self.system_size + 1, self.system_size + 1)
        # Lindeby definition
        # low = np.array([0, 0, 0, 0])
        # high = np.array([1, self.system_size, self.system_size, 3])
        # self.action_space = gym.spaces.Box(low, high)
        # self.observation_space = gym.spaces.Box(0, 1, [2, self.system_size, self.system_size])

        # TODO:
        # How to define the surface code matrix?

        # Look at Sweke code, they worked on the same surface code representation
        self.qubits = np.zeros((self.system_size, self.system_size), dtype=np.uint8)

        syndrome_size = self.system_size + 1
        self.syndrome_matrix = np.zeros(
            (2 * self.stack_depth, syndrome_size, syndrome_size), dtype=np.uint8
        )

        self.state = self.syndrome_matrix
        self.next_state = self.state

        # Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3
        self.rule_table = np.array(
            ([0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]), dtype=np.uint8
        )

        self.ground_state = True

    def step(self, action):
        """
        Apply a pauli operator to a qubit on the surface code.

        Parameters
        ==========
        action: (1, d, d, # operators) array defining x- & y-coordinates and operator type

        Returns
        =======
        state: (2, m, m) stacked syndrome arrays
        reward: int, reward for given action
        terminal: bool, determines if it is terminal state or not
        {}: empty dictionary, for conformity reasons #TODO or is it?
        """

        row = action[1]
        col = action[2]
        add_operator = action[3]

        old_operator = self.qubits[row, col]
        new_operator = self.rule_table[old_operator, add_operator]
        self.qubits[row, col] = new_operator

        self.next_state = self.create_syndrome_output(self.qubits)

        reward = self.get_reward()
        self.state = self.next_state

        terminal = self.is_terminal(self.state)

        return self.state, reward, terminal, {}

    def reset(self, p_error=None, p_msmt=None):
        pass

    def create_syndrome_output(self, qubits):
        """
        Infer the true syndrome output (w/o measurement errors)
        from the qubit matrix.
        Perform this for one slice.

        Parameters
        ==========
        qubits: (d, d) array containing the net operation performed on each qubit

        Returns
        =======
        syndrome: (d+1, d+1) array embedding vertices and plaquettes
        """
        syndrome = np.zeros_like(self.syndrome_matrix)

        qubits = np.pad(qubits, ((1, 0), (1, 0)), 'constant', constant_values=0)

        x = (qubits == 1).astype(np.uint8)
        y = (qubits == 2).astype(np.uint8)
        z = (qubits == 3).astype(np.uint8)
        assert x.shape == qubits.shape
        assert y.shape == qubits.shape
        assert z.shape == qubits.shape

        x_shifted_left = np.roll(x, -1, axis=1)
        x_shifted_up = np.roll(x, -1, axis=0)
        x_shifted_ul = np.roll(x_shifted_up, -1, axis=1) # shifted up and left

        z_shifted_left = np.roll(z, -1, axis=1)
        z_shifted_up = np.roll(z, -1, axis=0)
        z_shifted_ul = np.roll(z_shifted_up, -1, axis=1)

        y_shifted_left = np.roll(y, -1, axis=1)
        y_shifted_up = np.roll(y, -1, axis=0)
        y_shifted_ul = np.roll(y_shifted_up, -1, axis=1)
        
        # X = shaded = vertex
        syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * self.vertex_mask
        syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * self.vertex_mask

        # Z = blank = plaquette
        syndrome += (z + z_shifted_up + z_shifted_left + z_shifted_ul) * self.plaquette_mask
        syndrome += (y + y_shifted_up + y_shifted_left + y_shifted_ul) * self.plaquette_mask

        assert syndrome.shape == (self.system_size + 1, self.system_size + 1)

        syndrome = syndrome % 2 # we can only measure parity, hence only odd number of errors per syndrome
        return syndrome

    def get_reward(self):
        pass

    def is_terminal(self, state):
        pass


if __name__ == "__main__":
    sc = SurfaceCode()
