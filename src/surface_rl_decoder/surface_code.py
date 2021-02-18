"""
Define the Environment for decoding of the quantum surface code
to use it in reinforcement learning.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from iniparser import Config
from .syndrome_masks import vertex_mask, plaquette_mask
from .surface_code_util import (
    check_final_state,
    perform_action,
    is_terminal,
    RULE_TABLE,
    MAX_ACTIONS,
)


class SurfaceCode(gym.Env):
    """
    Description:
        Defines the rotated surface code for qubit encoding.

        New in this iteration of the project:
            The state now consists of multiple layers of syndrome slices.
            These layers correspond to different time steps, with the oldest time step at index 0.
            The resulting syndrome stack has a fixed height h
            (i.e. a fixed number of time steps that we keep track of).
            There might be the possibility that the actual error chain is shorter than the
            stack height.
            In this case, say with k faulty time slices, where k < h, only the k latest slices
            possess non-zero entries. Slices 0 up to h-k are all zero in this case.

            An action will act on one qubit throughout the whole stack.
            The goal of the decoder should be to get rid of all qubit erros by
            the latest slice (i.e. at index h-1).
            The decoder should learn to detect measurement errors and in such a case
            rightfully ignore those.
            Hence, it could happen that the latest slice contains errors of the
            measurement kind after the decoder has finished its job; this would be okay.

            Define:
            h = stack depth, i.e. number of time steps / time slices in the stack
            d = code distance


            # TODO need to keep track of measurement errors separately,
            # so that we know at prediction time
            # which errors are real.

    Actions:
        An action can be a Pauli X, Y, Z, or Identity on any qubit on the surface.

    Reward:

    Episode Termination:
        #TODO Either if the agent decides that it is terminated or if the last remaining
        # surface is error free.

    """

    def __init__(self):
        """
        Initialize Surface Code environment.
        Loads configuration via config-env-parsers, therefore we
        need either a config.ini file or constants saved as env variables.
        """

        cfg = Config()
        _config = cfg.scan(".", True).read()
        self.config = cfg.config_rendered.get("config")

        env_config = self.config.get("env")

        self.system_size = int(env_config.get("size"))
        self.syndrome_size = self.system_size + 1
        self.min_qbit_errors = int(env_config.get("min_qbit_err"))
        self.p_error = float(env_config.get("p_error"))
        self.p_msmt = float(env_config.get("p_msmt"))
        self.stack_depth = int(env_config.get("stack_depth"))
        self.error_channel = env_config.get("error_channel")

        # Sweke definition
        self.num_actions = 3 * self.system_size ** 2 + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.completed_actions = np.zeros(self.num_actions, np.uint8)

        self.volume_depth = 3  # what is this? TODO # possibly deprecated
        self.n_action_layers = (
            3  # what is this? In the case with Y errors, this is 3 TODO
        )

        # observation space should correspond to the shape
        # of vertex- and plaquette-representation
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.stack_depth, self.syndrome_size, self.syndrome_size),
            dtype=np.uint8,
        )

        # imported from file
        self.vertex_mask = np.tile(vertex_mask, (self.stack_depth, 1, 1))
        self.plaquette_mask = np.tile(plaquette_mask, (self.stack_depth, 1, 1))
        assert self.vertex_mask.shape == (
            self.stack_depth,
            self.system_size + 1,
            self.system_size + 1,
        ), vertex_mask.shape
        assert self.plaquette_mask.shape == (
            self.stack_depth,
            self.system_size + 1,
            self.system_size + 1,
        ), plaquette_mask.shape

        # TODO:
        # How to define the surface code matrix?
        # Idea: define both plaquettes and vertices on a (d+1, d+1) matrix
        # https://app.diagrams.net/#G1Ppj6myKPwCny7QeFz9cNq2TC_h6fwkn6

        # Look at Sweke code, they worked on the same surface code representation
        self.qubits = np.zeros(
            (self.stack_depth, self.system_size, self.system_size), dtype=np.uint8
        )

        # define syndrome matrix
        self.state = np.zeros(
            (self.stack_depth, self.syndrome_size, self.syndrome_size), dtype=np.uint8
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                self.stack_depth,
                self.syndrome_size,
                self.syndrome_size,
            ),
            dtype=np.uint8,
        )

        self.actual_errors = self.qubits
        self.next_state = self.state

        # Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3
        self.rule_table = RULE_TABLE

        # container to save action history
        self.actions = np.zeros((MAX_ACTIONS, 3), dtype=np.uint8)
        self.current_action_index = 0

        self.ground_state = True

    def step(self, action):
        """
        Apply a pauli operator to a qubit on the surface code with code distance d.

        Parameters
        ==========
        action: tuple containing (None, x-coordinate, y-coordinate, pauli operator),
            defining x- & y-coordinates and operator type

        Returns
        =======
        state: (d+1, d+1) stacked syndrome arrays
        reward: int, reward for given action
        terminal: bool, determines if it is terminal state or not
        {}: empty dictionary, for conformity reasons #TODO or is it?
        """
        self.actions[self.current_action_index] = action[-3:]
        self.current_action_index += 1

        # execute operation throughout the stack
        self.qubits = perform_action(self.qubits, action)

        self.next_state = self.create_syndrome_output(self.qubits)

        reward = self.get_reward(action)
        self.state = self.next_state
        terminal = is_terminal(action)

        return self.state, reward, terminal, {}

    def generate_qubit_x_error(self):
        """
        Generate only X errors on the ubit grid.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        In those elements where the random value was below p_error, an X operation is performed.

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error = (uniform_random_vector < self.p_error).astype(np.uint8)

        return error

    def generate_qubit_iidxz_error(self):
        """
        Generate X and Z qubit errors independent of each other on one slice in vectorized form.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        Then, the actual error operation out of the set (X, Y, Z) = (1, 2, 3) is chosen
        for each element.
        However, only in those elements where the random value was below p_error,
        the operation is saved by multiplying the operaton matrix with the mask array.

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask_x = (uniform_random_vector < self.p_error).astype(np.uint8)
        x_err = np.ones(shape, dtype=np.uint8)
        x_err = np.multiply(x_err, error_mask_x)

        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask_z = (uniform_random_vector < self.p_error).astype(np.uint8)
        z_err = np.ones(shape, dtype=np.uint8) * 3
        z_err = np.multiply(z_err, error_mask_z)

        err = x_err + z_err
        error = np.where(
            err > 3, 2, err
        )  # where x and z errors add up, write a 2 instead

        return error

    def generate_qubit_dp_error(self):
        """
        Generate depolarizing qubit errors on one slice in vectorized form.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        Then, the actual error operation out of the set (X, Y, Z) = (1, 2, 3) is chosen
        for each element.
        However, only in those elements where the random value was below p_error,
        the operation is saved by multiplying the operaton matrix with the mask array.

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask = (uniform_random_vector < self.p_error).astype(np.uint8)

        error_operation = np.random.randint(1, 4, shape, dtype=np.uint8)
        error = np.multiply(error_mask, error_operation)
        error = error.astype(np.uint8)

        return error

    def generate_qubit_error(self, error_channel=None):
        """
        Wrapper function to create qubit errors vis the corret error channel.

        Parameters
        ==========
        error_channel: (optional) either "dp", "x", "iidxz" denoting the error channel of choice

        Returns
        =======
        error: (d, d) array of qubits with occasional error operations
        """

        if error_channel is None:
            error_channel = self.error_channel

        if error_channel in ("x", "X"):
            error = self.generate_qubit_x_error()
        elif error_channel in ("dp", "DP"):
            error = self.generate_qubit_dp_error()
        elif error_channel in ("iidxz", "IIDXZ"):
            error = self.generate_qubit_iidxz_error()
        else:
            raise Exception(f"Error! error channel {error_channel} not supported.")

        return error

    def generate_qubit_error_stack(self, error_channel="dp"):
        """
        Create a whole stack of qubits which act as the time evolution
        of the surface code through time.
        Each higher layer in the stack is dependent on the layers below, carrying over
        old errors and potentially introducing new errors proportional to the error probability.

        Parameters
        ==========
        error_channel: (optional) either "dp", "x", "iidxz" denoting the error channel of choice

        Returns
        =======
        error_stack: (h, d, d) array of qubits; qubit slices through time
            with occasional error operations
        """
        # TODO: can extend this function to also support error stacks
        # where errors occur only after a certain time
        error_stack = np.zeros(
            (self.stack_depth, self.system_size, self.system_size), dtype=np.uint8
        )
        base_error = self.generate_qubit_error(error_channel=error_channel)

        error_stack[0, :, :] = base_error
        for height in range(1, self.stack_depth):
            new_error = self.generate_qubit_error(error_channel=error_channel)

            # filter where errors have actually occured with np.where()
            nonzero_idx = np.where(base_error != 0)

            # TODO: could possibly optimize this
            for row in nonzero_idx[0]:
                for col in nonzero_idx[1]:
                    old_operator = base_error[row, col]
                    new_error[row, col] = self.rule_table[
                        old_operator, new_error[row, col]
                    ]

            error_stack[height, :, :] = new_error
            base_error = new_error

        return error_stack

    def generate_measurement_error(self, true_syndrome):
        """
        Introduce random measurement errors on the syndrome.

        Works both on slices and on stacks.

        Parameters
        ==========
        true_syndrome: (d, d) or (h, d, d) array of syndrome embedding

        Returns
        =======
        faulty_syndrome: (d, d) or (h, d, d) array of syndrome with occasional erroneous syndrome
            measurements
        """
        shape = true_syndrome.shape

        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask = (uniform_random_vector < self.p_msmt).astype(np.uint8)

        # take into account positions of vertices and plaquettes
        error_mask = np.multiply(error_mask, np.add(plaquette_mask, vertex_mask))
        # TODO: could just save the error_mask here to keep track of
        # where real errors are and where msmt errors are
        # Or one could save the error array, output from generate_qubit_error()

        # where an error occurs, flip the true syndrome measurement
        faulty_syndrome = np.where(error_mask > 0, 1 - true_syndrome, true_syndrome)

        return faulty_syndrome

    def reset(self, error_channel="dp"):
        """
        Reset the environment and generate new qubit and syndrome stacks with errors.

        Returns
        =======
        state: (d+1, d+1) stacked syndrome arrays
        """

        self.ground_state = True

        self.qubits = np.zeros(
            (self.stack_depth, self.system_size, self.system_size), dtype=np.uint8
        )
        self.state = np.zeros(
            (self.stack_depth, self.syndrome_size, self.syndrome_size), dtype=np.uint8
        )
        self.next_state = np.zeros(
            (self.stack_depth, self.syndrome_size, self.syndrome_size), dtype=np.uint8
        )

        self.actions = np.zeros_like(self.actions)

        # TODO: implement function to generate minimum number of errors
        while self.qubits.sum() == 0:
            self.qubits = self.generate_qubit_error_stack(error_channel=error_channel)
            true_syndrome = self.create_syndrome_output_stack(self.qubits)
            self.state = self.generate_measurement_error(true_syndrome)

        self.actual_errors = self.qubits
        return self.state

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

        # make sure it is only one slice
        if len(qubits.shape) != 2:
            if len(qubits.shape) == 3:
                assert qubits.shape[0] == 1, qubits.shape

        # pad with ((one row above, zero rows below), (one row to the left, zero rows to the right))
        qubits = np.pad(qubits, ((1, 0), (1, 0)), "constant", constant_values=0)

        # pylint: disable=invalid-name
        x = (qubits == 1).astype(np.uint8)
        y = (qubits == 2).astype(np.uint8)
        z = (qubits == 3).astype(np.uint8)
        assert x.shape == qubits.shape, x.shape
        assert y.shape == qubits.shape, y.shape
        assert z.shape == qubits.shape, z.shape

        x_shifted_left = np.roll(x, -1, axis=1)
        x_shifted_up = np.roll(x, -1, axis=0)
        x_shifted_ul = np.roll(x_shifted_up, -1, axis=1)  # shifted up and left

        z_shifted_left = np.roll(z, -1, axis=1)
        z_shifted_up = np.roll(z, -1, axis=0)
        z_shifted_ul = np.roll(z_shifted_up, -1, axis=1)

        y_shifted_left = np.roll(y, -1, axis=1)
        y_shifted_up = np.roll(y, -1, axis=0)
        y_shifted_ul = np.roll(y_shifted_up, -1, axis=1)

        # X = shaded = vertex
        syndrome = (
            x + x_shifted_up + x_shifted_left + x_shifted_ul
        ) * self.vertex_mask[0]
        syndrome += (
            y + y_shifted_up + y_shifted_left + y_shifted_ul
        ) * self.vertex_mask[0]

        # Z = blank = plaquette
        syndrome += (
            z + z_shifted_up + z_shifted_left + z_shifted_ul
        ) * self.plaquette_mask[0]
        syndrome += (
            y + y_shifted_up + y_shifted_left + y_shifted_ul
        ) * self.plaquette_mask[0]

        assert syndrome.shape == (
            self.system_size + 1,
            self.system_size + 1,
        ), syndrome.shape

        syndrome = (
            syndrome % 2
        )  # we can only measure parity, hence only odd number of errors per syndrome
        return syndrome

    def create_syndrome_output_stack(self, qubits):
        """
        Infer the true syndrome output (w/o measurement errors)
        from the qubit matrix.

        d: code distance
        h: stack depth/height

        Parameters
        ==========
        qubits: (h, d, d) array containing the net operation performed on each qubit

        Returns
        =======
        syndrome: (h, d+1, d+1) array embedding vertices and plaquettes
        """
        # regard this as pseudo code
        # just writing down ideas

        # pad with (
        #   (nothing along time axis),
        #   (one row above, zero rows below),
        #   (one row to the left, zero rows to the right)
        # )
        qubits = np.pad(qubits, ((0, 0), (1, 0), (1, 0)), "constant", constant_values=0)

        # pylint: disable=invalid-name
        x = (qubits == 1).astype(np.uint8)
        y = (qubits == 2).astype(np.uint8)
        z = (qubits == 3).astype(np.uint8)
        assert x.shape == qubits.shape
        assert y.shape == qubits.shape
        assert z.shape == qubits.shape

        x_shifted_left = np.roll(x, -1, axis=2)
        x_shifted_up = np.roll(x, -1, axis=1)
        x_shifted_ul = np.roll(x_shifted_up, -1, axis=2)  # shifted up and left

        z_shifted_left = np.roll(z, -1, axis=2)
        z_shifted_up = np.roll(z, -1, axis=1)
        z_shifted_ul = np.roll(z_shifted_up, -1, axis=2)

        y_shifted_left = np.roll(y, -1, axis=2)
        y_shifted_up = np.roll(y, -1, axis=1)
        y_shifted_ul = np.roll(y_shifted_up, -1, axis=2)

        # X = shaded = vertex
        syndrome = (x + x_shifted_up + x_shifted_left + x_shifted_ul) * self.vertex_mask
        syndrome += (
            y + y_shifted_up + y_shifted_left + y_shifted_ul
        ) * self.vertex_mask

        # Z = blank = plaquette
        syndrome += (
            z + z_shifted_up + z_shifted_left + z_shifted_ul
        ) * self.plaquette_mask
        syndrome += (
            y + y_shifted_up + y_shifted_left + y_shifted_ul
        ) * self.plaquette_mask

        assert syndrome.shape == (
            self.stack_depth,
            self.system_size + 1,
            self.system_size + 1,
        ), syndrome.shape

        syndrome = (
            syndrome % 2
        )  # we can only measure parity, hence only odd number of errors per syndrome
        return syndrome

    def get_reward(self, action):
        """
        Calculate reward for each action.

        For regular operations, no reward is given.
        Only when the agent declares the state to be terminal do
        we distribute rewards or penalties.

        The strategy is to give negative reward if the surface code
        is not even in the ground state.
        If there are still errors left on the surface, this gives a
        negative reward as well, but of smaller magnitude.
        If after the correction there are only trivial loops left on the surface,
        the agent earns high positive reward.

        Parameters
        ==========
        action: tuple containing (None, x-coordinate, y-coordinate, pauli operator),
            defining x- & y-coordinates and operator type

        Returns
        =======
        reward (int)
        """
        row = action[1]
        col = action[2]
        operator = action[3]

        if operator in (1, 2, 3):
            return 0

        # TODO: need to check for repeated actions

        # assume action "terminal" was chosen
        final_state, is_ground_state = check_final_state(
            self.actual_errors, self.actions
        )

        # not in the ground state; meaning the agent
        # performed a logical operation by accident
        if not is_ground_state:
            return -1000

        # ground state but still some qubit errors persist
        if final_state.sum() != 0:
            return -100

        # ground state and all qubit errors have been corrected
        # great success
        return 1000

    def render(self, mode="human"):
        """
        Not supported yet. Needed for conformity with abstract base class.
        """
        # pylint: disable=unnecessary-pass

        # slider location
        location = plt.axes([])
        slider = Slider(location, "Time", 0, self.stack_depth, valinit=0)

        def update_depth(val):
            idx = int(round(slider.val))
            img.set_data(self.qubits[idx, :, :])

        slider.on_changed(update_depth)

        pass


if __name__ == "__main__":
    sc = SurfaceCode()
