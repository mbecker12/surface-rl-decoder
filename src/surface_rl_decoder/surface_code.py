"""
Define the Environment for decoding of the quantum surface code
to use it in reinforcement learning.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Wedge, Rectangle
from iniparser import Config
from .syndrome_masks import vertex_mask, plaquette_mask
from .surface_code_util import (
    NON_TRIVIAL_LOOP_REWARD,
    SOLVED_EPISODE_REWARD,
    SYNDROME_LEFT_REWARD,
    check_final_state,
    create_syndrome_output_stack,
    perform_action,
    copy_array_values,
    RULE_TABLE,
    MAX_ACTIONS,
    TERMINAL_ACTION,
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

    Actions:
        An action can be a Pauli X, Y, Z, or Identity on any qubit on the surface.

    Reward:

    Episode Termination:
        #TODO Either if the agent decides that it is terminated or if the last remaining
        # surface is error free.
        # Or possibly if the agent repeats an action.

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

        # How we define the surface code matrix:
        # Idea: define both plaquettes and vertices on a (d+1, d+1) matrix
        # https://app.diagrams.net/#G1Ppj6myKPwCny7QeFz9cNq2TC_h6fwkn6

        # implement rotated surface code
        # from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.041031 Fig. 4
        self.qubits = np.zeros(
            (self.stack_depth, self.system_size, self.system_size), dtype=np.uint8
        )

        # define syndrome matrix
        self.state = np.zeros(
            (self.stack_depth, self.syndrome_size, self.syndrome_size), dtype=np.uint8
        )

        # actual initial qubit configuration, physical errors
        self.actual_errors = np.zeros_like(self.qubits)
        self.next_state = np.zeros_like(self.state)

        # syndrome measurement errors
        self.syndrome_errors = np.zeros_like(self.state)

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
        {}: empty dictionary, for conformity reasons; supposed to be used for info
        """
        self.actions[self.current_action_index] = action[-3:]
        self.current_action_index += 1

        # execute operation throughout the stack
        terminal = action[-1] == TERMINAL_ACTION
        reward = self.get_reward(action)

        if terminal:
            return self.state, reward, terminal, {}

        self.qubits = perform_action(self.qubits, action)

        syndrome = create_syndrome_output_stack(
            self.qubits, self.vertex_mask, self.plaquette_mask
        )

        self.next_state = np.logical_xor(syndrome, self.syndrome_errors)
        self.state = self.next_state

        # if we reach the action history limit
        # force the episode to be over and determine
        # the reward based on the state after the latest action
        if self.current_action_index == MAX_ACTIONS:
            reward = self.get_reward(action=(-1, -1, TERMINAL_ACTION))
            terminal = True

        return self.state, reward, terminal, {}

    def generate_qubit_x_error(self, min_n_errors=-1):
        """
        Generate only X errors on the ubit grid.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        In those elements where the random value was below p_error, an X operation is performed.

        Parameters
        ==========
        min_n_errors: (optional) (int) minimum number of qubit errors that
            should be introduced in this layer

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask = (uniform_random_vector < self.p_error).astype(np.uint8)

        if min_n_errors > 0 and error_mask.sum() < min_n_errors:
            idx = np.random.randint(0, self.system_size, size=(2, min_n_errors))
            error_mask[idx[0], idx[1]] = 1

        return error_mask

    def generate_qubit_iidxz_error(self, min_n_errors=-1):
        """
        Generate X and Z qubit errors independent of each other on one slice in vectorized form.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        Then, the actual error operation out of the set (X, Y, Z) = (1, 2, 3) is chosen
        for each element.
        However, only in those elements where the random value was below p_error,
        the operation is saved by multiplying the operaton matrix with the mask array.

        Parameters
        ==========
        min_n_errors: (optional) (int) minimum number of qubit errors that
            should be introduced in this layer

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask_x = (uniform_random_vector < self.p_error).astype(np.uint8)

        if min_n_errors > 0:
            min_x_errors = min_n_errors // 2 + min_n_errors % 2
            min_z_errors = min_n_errors // 2

            if error_mask_x.sum() < min_x_errors:
                idx = np.random.randint(0, self.system_size, size=(2, min_n_errors))
                error_mask_x[idx[0], idx[1]] = 1

        x_err = np.ones(shape, dtype=np.uint8)
        x_err = np.multiply(x_err, error_mask_x)

        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask_z = (uniform_random_vector < self.p_error).astype(np.uint8)

        if min_n_errors > 0:
            if error_mask_z.sum() < min_z_errors:
                idx = np.random.randint(0, self.system_size, size=(2, min_n_errors))
                error_mask_z[idx[0], idx[1]] = 1

        z_err = np.ones(shape, dtype=np.uint8) * 3
        z_err = np.multiply(z_err, error_mask_z)

        err = x_err + z_err
        error = np.where(
            err > 3, 2, err
        )  # where x and z errors add up, write a 2 instead

        return error

    def generate_qubit_dp_error(self, min_n_errors=-1):
        """
        Generate depolarizing qubit errors on one slice in vectorized form.

        First, create a matrix with random values in [0, 1] and compare each element
        to the error probability.
        Then, the actual error operation out of the set (X, Y, Z) = (1, 2, 3) is chosen
        for each element.
        However, only in those elements where the random value was below p_error,
        the operation is saved by multiplying the operaton matrix with the mask array.

        Parameters
        ==========
        min_n_errors: (optional) (int) minimum number of qubit errors that
            should be introduced in this layer

        Returns
        =======
        error: (d, d) array containing error operations on a qubit grid
        """
        shape = (self.system_size, self.system_size)
        uniform_random_vector = np.random.uniform(0.0, 1.0, shape)
        error_mask = (uniform_random_vector < self.p_error).astype(np.uint8)

        if min_n_errors > 0 and error_mask.sum() < min_n_errors:
            idx = np.random.randint(0, self.system_size, size=(2, min_n_errors))
            error_mask[idx[0], idx[1]] = 1

        error_operation = np.random.randint(1, 4, shape, dtype=np.uint8)
        error = np.multiply(error_mask, error_operation)
        error = error.astype(np.uint8)

        return error

    def generate_qubit_error(self, error_channel=None, min_n_errors=-1):
        """
        Wrapper function to create qubit errors vis the corret error channel.

        Parameters
        ==========
        error_channel: (optional) either "dp", "x", "iidxz" denoting the error channel of choice
        min_n_errors: (optional) (int) minimum number of qubit errors that
            should be introduced in this layer

        Returns
        =======
        error: (d, d) array of qubits with occasional error operations
        """

        if error_channel is None:
            error_channel = self.error_channel

        if error_channel in ("x", "X"):
            error = self.generate_qubit_x_error(min_n_errors=min_n_errors)
        elif error_channel in ("dp", "DP"):
            error = self.generate_qubit_dp_error(min_n_errors=min_n_errors)
        elif error_channel in ("iidxz", "IIDXZ"):
            error = self.generate_qubit_iidxz_error(min_n_errors=min_n_errors)
        else:
            raise Exception(f"Error! error channel {error_channel} not supported.")

        return error

    def generate_qubit_error_stack(self, error_channel="dp", min_n_errors=-1):
        """
        Create a whole stack of qubits which act as the time evolution
        of the surface code through time.
        Each higher layer in the stack is dependent on the layers below, carrying over
        old errors and potentially introducing new errors proportional to the error probability.

        Parameters
        ==========
        error_channel: (optional) either "dp", "x", "iidxz" denoting the error channel of choice
        min_n_errors: (optional) (int) minimum number of qubit errors that
            should be introduced in the first layer

        Returns
        =======
        error_stack: (h, d, d) array of qubits; qubit slices through time
            with occasional error operations
        """

        error_stack = np.zeros(
            (self.stack_depth, self.system_size, self.system_size), dtype=np.uint8
        )
        base_error = self.generate_qubit_error(
            error_channel=error_channel, min_n_errors=min_n_errors
        )

        error_stack[0, :, :] = base_error
        for height in range(1, self.stack_depth):
            # don't use min_n_errors here;
            # at this point, we merely want to build on top of the fround layer
            new_error = self.generate_qubit_error(error_channel=error_channel)

            # filter where errors have actually occured with np.where()
            nonzero_idx = np.where(np.logical_or(new_error, base_error))

            for row, col in zip(*nonzero_idx):
                old_operator = base_error[row, col]
                new_error[row, col] = self.rule_table[old_operator, new_error[row, col]]

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
        self.actual_errors = np.zeros_like(self.qubits)
        self.state = np.zeros(
            (self.stack_depth, self.syndrome_size, self.syndrome_size), dtype=np.uint8
        )
        self.next_state = np.zeros_like(self.state)

        self.actions = np.zeros_like(self.actions)
        self.syndrome_errors = np.zeros_like(self.state)

        if self.p_msmt > 0 and self.p_error > 0:
            self.actual_errors = self.generate_qubit_error_stack(
                error_channel=error_channel, min_n_errors=self.min_qbit_errors
            )

        true_syndrome = create_syndrome_output_stack(
            self.actual_errors, self.vertex_mask, self.plaquette_mask
        )
        self.state = self.generate_measurement_error(true_syndrome)
        # save the introduced syndrome errors by checking the difference
        # between the true syndrome from qubit errors
        # and the updated syndrome with measurement errors
        self.syndrome_errors = np.logical_xor(self.state, true_syndrome)

        self.qubits = copy_array_values(self.actual_errors)
        return self.state

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
        _, _, operator = action[-3:]

        if operator in (1, 2, 3):
            return 0

        # assume action "terminal" was chosen
        actual_errors = copy_array_values(self.actual_errors)
        _, self.ground_state, (n_syndromes, n_loops) = check_final_state(
            actual_errors, self.actions, self.vertex_mask, self.plaquette_mask
        )

        if self.ground_state:
            return SOLVED_EPISODE_REWARD

        # not in the ground state; meaning the agent
        # performed a logical operation by accident
        return n_loops * NON_TRIVIAL_LOOP_REWARD + n_syndromes * SYNDROME_LEFT_REWARD

    def render(self, mode="human", block=True):
        """
        Visualize the environment.

        Creates a plot of the qubit grid with a slider
        to visualize time evolution of qubits.

        Parameters
        ==========
        mode: (str) render mode
        block: (bool) whether or not pyplot should show the image
        """

        if mode == "human":
            markersize_qubit = 15
            markersize_excitation = 7
            markersize_symbols = 7
            linewidth = 2

            vertex_matrix = np.multiply(
                np.logical_xor(self.state, self.syndrome_errors), self.vertex_mask
            )
            plaquette_matrix = np.multiply(
                np.logical_xor(self.state, self.syndrome_errors), self.plaquette_mask
            )

            vertex_msmt_error_matrix = np.multiply(
                self.syndrome_errors, self.vertex_mask
            )

            plaquette_msmt_error_matrix = np.multiply(
                self.syndrome_errors, self.plaquette_mask
            )

            self.setup_qubit_grid(
                markersize_qubit=markersize_qubit,
                linewidth=linewidth,
            )

            # slider location
            location = plt.axes([0.1, 0, 0.8, 0.05])
            slider = Slider(
                location,
                "Layer",
                0,
                self.stack_depth - 1,
                valinit=0,
                valfmt="%i",
            )

            # pylint: disable=unused-argument
            def update_slider(val):
                idx = int(slider.val)

                ax = self.setup_qubit_grid(
                    markersize_qubit=markersize_qubit,
                    linewidth=linewidth,
                )

                # proceed to draw faulty qubits and activated syndromes
                x_error_idx = np.where(self.qubits[idx] == 1)
                y_error_idx = np.where(self.qubits[idx] == 2)
                z_error_idx = np.where(self.qubits[idx] == 3)

                ax.plot(
                    x_error_idx[1],
                    -x_error_idx[0],
                    "o",
                    color="r",
                    label="x error",
                    markersize=markersize_qubit,
                )
                ax.plot(
                    x_error_idx[1],
                    -x_error_idx[0],
                    "o",
                    color="black",
                    markersize=markersize_symbols,
                    marker=r"$X$",
                )
                ax.plot(
                    y_error_idx[1],
                    -y_error_idx[0],
                    "o",
                    color="blueviolet",
                    label="y error",
                    markersize=markersize_qubit,
                )
                ax.plot(
                    y_error_idx[1],
                    -y_error_idx[0],
                    "o",
                    color="black",
                    markersize=markersize_symbols,
                    marker=r"$Y$",
                )
                ax.plot(
                    z_error_idx[1],
                    -z_error_idx[0],
                    "o",
                    color="b",
                    label="z error",
                    markersize=markersize_qubit,
                )
                ax.plot(
                    z_error_idx[1],
                    -z_error_idx[0],
                    "o",
                    color="black",
                    markersize=markersize_symbols,
                    marker=r"$Z$",
                )

                vertex_idx = np.where(vertex_matrix[idx])
                plaquette_idx = np.where(plaquette_matrix[idx])
                ax.plot(
                    vertex_idx[1] - 0.5,
                    -vertex_idx[0] + 0.5,
                    "o",
                    color="blue",
                    label="charge",
                    markersize=markersize_excitation,
                )
                ax.plot(
                    plaquette_idx[1] - 0.5,
                    -plaquette_idx[0] + 0.5,
                    "o",
                    color="red",
                    label="flux",
                    markersize=markersize_excitation,
                )

                # visualize measurement errors
                vertex_msmt_idx = np.where(vertex_msmt_error_matrix[idx])
                plaquette_msmt_idx = np.where(plaquette_msmt_error_matrix[idx])
                ax.plot(
                    vertex_msmt_idx[1] - 0.5,
                    -vertex_msmt_idx[0] + 0.5,
                    "o",
                    color="black",
                    label="charge",
                    mfc="none",
                    markersize=markersize_excitation + 2,
                    markeredgewidth=2,
                )
                ax.plot(
                    plaquette_msmt_idx[1] - 0.5,
                    -plaquette_msmt_idx[0] + 0.5,
                    "o",
                    color="black",
                    label="flux",
                    mfc="none",
                    markersize=markersize_excitation + 2,
                    markeredgewidth=2,
                )

            slider.on_changed(update_slider)
            plt.show(block=block)
        else:
            raise Exception(f"Error! Mode {mode} not supported!")

    def setup_qubit_grid(
        self,
        markersize_qubit=15,
        linewidth=2,
    ):
        """
        Prepare the visual representation of the qubit grid,
        including qubits, vertex syndromes, plaquette syndromes.
        """

        x_line = np.linspace(0, self.system_size - 1, self.system_size - 1)
        x_arr = range(self.system_size)
        x_grid, y_grid = np.meshgrid(x_arr, x_arr)
        x_line, y_line = np.meshgrid(x_arr, x_line)

        # provide a new label everytime, to prevent pyplot from using
        # the same instance of ax everytima
        ax = plt.subplot(111, label=str(np.random.rand()))

        # draw grid
        ax.plot(x_line, -y_line, "black", linewidth=linewidth)
        ax.plot(y_line, -x_line, "black", linewidth=linewidth)
        # draw qubits
        ax.plot(
            x_grid,
            -y_grid,
            "o",
            color="black",
            markerfacecolor="white",
            markersize=markersize_qubit + 1,
        )
        ax.plot(
            x_grid,
            -y_grid,
            "o",
            color="black",
            markerfacecolor="white",
            markersize=markersize_qubit + 1,
        )

        # draw vertex syndromes
        for i in range(0, self.system_size - 1):
            if i % 2 == 0:
                for j in range(1, self.system_size - 1, 2):
                    rect = Rectangle((i, -j), 1, 1, lw=linewidth, fc="orange")
                    ax.add_artist(rect)

            else:
                for j in range(2, self.system_size, 2):
                    rect = Rectangle((i, -j), 1, 1, lw=linewidth, fc="orange")
                    ax.add_artist(rect)

        # draw boundary syndromes
        for i in range(0, self.system_size - 1, 2):
            w_upper = Wedge(
                (i + 0.5, 0.0), 0.5, 0, 180, fc="white", ec="black", lw=linewidth
            )
            w_lower = Wedge(
                (i + 1.5, -self.system_size + 1),
                0.5,
                180,
                0,
                fc="white",
                ec="black",
                lw=linewidth,
            )
            ax.add_artist(w_lower)
            ax.add_artist(w_upper)

        for i in range(0, self.system_size - 1, 2):
            w_right = Wedge(
                (self.system_size - 1, -i - 0.5),
                0.5,
                270,
                90,
                fc="orange",
                ec="black",
                lw=linewidth,
            )
            w_left = Wedge(
                (0.0, -i - 1.5), 0.5, 90, 270, fc="orange", ec="black", lw=linewidth
            )
            ax.add_artist(w_right)
            ax.add_artist(w_left)

        ax.set_ylim(-self.system_size, 1)
        ax.set_xlim(-1, self.system_size)
        ax.set_title(f"Surface Code, d={self.system_size}, h={self.stack_depth}")
        return ax


if __name__ == "__main__":
    sc = SurfaceCode()
