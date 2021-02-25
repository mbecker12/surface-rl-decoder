from src.surface_rl_decoder.surface_code import SurfaceCode
from src.surface_rl_decoder.surface_code_util import (
    NON_TRIVIAL_LOOP_REWARD,
    SYNDROME_LEFT_REWARD,
    SOLVED_EPISODE_REWARD,
    TERMINAL_ACTION,
    copy_array_values,
    create_syndrome_output_stack,
)
from tests.data_episode_test import (
    _actions,
)


def test_successful_episode(seed_surface_code, configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    # pylint: disable=duplicate-code
    sc = SurfaceCode()
    seed_surface_code(sc, 42, 0.1, 0.1, "dp")

    for action in _actions:
        sc.step(action)

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))

    assert terminal
    assert reward == SOLVED_EPISODE_REWARD

    restore_env(original_depth, original_size, original_error_channel)


def test_remaining_syndromes(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    sc.actual_errors[-1, 3, 4] = 1  # X error on the edge, triggers 1 plaquette
    sc.actual_errors[-1, 2, 1] = 3  # Z error in the bulk, triggers 2 vertices

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    assert reward == (1 + 2) * SYNDROME_LEFT_REWARD

    restore_env(original_depth, original_size, original_error_channel)


def test_remaining_syndrome(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    sc.actual_errors[-1, 3, 2] = 1  # X error in the bulk, triggers 2 plaquettes

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    assert reward == 1 * 2 * SYNDROME_LEFT_REWARD, sc.state[-1]

    restore_env(original_depth, original_size, original_error_channel)


def test_remaining_trivial_loops(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    # introduce a trivial loop in 5x5 code
    sc.actual_errors[-1, 3, 2] = 1
    sc.actual_errors[-1, 3, 3] = 1
    sc.actual_errors[-1, 2, 2] = 1
    sc.actual_errors[-1, 2, 3] = 1

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    # trivial loops introduce no syndrome and no logical operation
    assert reward == SOLVED_EPISODE_REWARD, sc.state[-1]

    restore_env(original_depth, original_size, original_error_channel)


def test_non_trivial_loop(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    # introduce a trivial loop in 5x5 code
    sc.actual_errors[-1, 0, 2] = 3
    sc.actual_errors[-1, 1, 2] = 3
    sc.actual_errors[-1, 2, 2] = 3
    sc.actual_errors[-1, 3, 2] = 3
    sc.actual_errors[-1, 4, 2] = 3

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    # the above configuration introduces a non-trivial loop
    # (in this case spanning 5 qubits)
    # and thus a logical operation
    assert reward == 5 * NON_TRIVIAL_LOOP_REWARD, (sc.state[-1], sc.qubits[-1])
    assert sc.state[-1].sum() == 0

    restore_env(original_depth, original_size, original_error_channel)


def test_remaining_syndromes_loop(configure_env, restore_env):
    """
    assemble qubit X errors in a loop around a vertex, thus triggering
    multiple vertex syndromes.
    """
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    # introduce 4 X errors in 5x5 code, around a plaquette
    sc.actual_errors[-1, 1, 2] = 1
    sc.actual_errors[-1, 1, 3] = 1
    sc.actual_errors[-1, 2, 2] = 1
    sc.actual_errors[-1, 2, 3] = 1

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    # the above configuration should introduce 4 syndromes
    assert reward == 4 * SYNDROME_LEFT_REWARD, sc.state[-1]

    restore_env(original_depth, original_size, original_error_channel)


def test_long_non_trivial_loops(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    # introduce a somewhat tilted non-trivial loop in 5x5 code
    sc.actual_errors[-1, 3, 0] = 1
    sc.actual_errors[-1, 3, 1] = 1
    sc.actual_errors[-1, 2, 2] = 1
    sc.actual_errors[-1, 1, 3] = 1
    sc.actual_errors[-1, 1, 4] = 1

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    # the above configuration introduces a non-trivial loop
    # and thus a logical operation
    assert reward == 5 * NON_TRIVIAL_LOOP_REWARD, sc.state[-1]

    restore_env(original_depth, original_size, original_error_channel)


def test_long_non_trivial_loops2(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.p_error = 0
    sc.p_msmt = 0
    sc.reset()

    # introduce a longer trivial loop in 5x5 code

    sc.actual_errors[-1, 0, 3] = 3
    sc.actual_errors[-1, 1, 2] = 3
    sc.actual_errors[-1, 2, 2] = 3
    sc.actual_errors[-1, 3, 3] = 3
    sc.actual_errors[-1, 4, 3] = 3

    sc.qubits = copy_array_values(sc.actual_errors)
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    _, reward, terminal, _ = sc.step(action=(-1, -1, TERMINAL_ACTION))
    assert terminal
    # the above configuration introduces a non-trivial loop
    # and thus a logical operation
    assert reward == 5 * NON_TRIVIAL_LOOP_REWARD, sc.state[-1]

    restore_env(original_depth, original_size, original_error_channel)
