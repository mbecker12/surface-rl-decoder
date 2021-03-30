import pytest
import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode
from src.surface_rl_decoder.surface_code_util import (
    SYNDROME_DIFF_REWARD,
    create_syndrome_output_stack,
)


def test_intermediate_reward_simplest(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.state = np.zeros((4, 6, 6), dtype=np.uint8)

    sc.qubits[:, 2, 2] = 3
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    action = (2, 2, 3)
    discount_factor = 0.75
    annealing_factor = 1.0
    _, reward, _, _ = sc.step(
        action,
        discount_intermediate_reward=discount_factor,
        annealing_intermediate_reward=annealing_factor,
    )

    assert pytest.approx(reward) == SYNDROME_DIFF_REWARD * (
        2 + 2 * discount_factor + 2 * discount_factor ** 2 + 2 * discount_factor ** 3
    )
    restore_env(original_depth, original_size, original_error_channel)


def test_intermediate_reward_halfway(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.state = np.zeros((4, 6, 6), dtype=np.uint8)
    sc.qubits[2:, 3, 1] = 1
    sc.state = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )

    action = (3, 1, 1)
    discount_factor = 0.75
    annealing_factor = 0.9
    _, reward, _, _ = sc.step(
        action,
        discount_intermediate_reward=discount_factor,
        annealing_intermediate_reward=annealing_factor,
    )

    assert pytest.approx(reward / annealing_factor) == SYNDROME_DIFF_REWARD * (
        2 + 2 * discount_factor - (2 * discount_factor ** 2 + 2 * discount_factor ** 3)
    )

    restore_env(original_depth, original_size, original_error_channel)


def test_reward_w_syndrome_error(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    sc.state = np.zeros((4, 6, 6), dtype=np.uint8)
    sc.syndrome_errors[1, 2, 3] = 1
    sc.syndrome_errors[3, 2, 2] = 1
    sc.syndrome_errors[2, 4, 3] = 1

    sc.qubits[:, 1, 1] = 3
    sc.qubits[1:, 3, 2] = 1
    true_syndrome = create_syndrome_output_stack(
        sc.qubits, sc.vertex_mask, sc.plaquette_mask
    )
    sc.state = np.logical_xor(true_syndrome, sc.syndrome_errors)

    action1 = (1, 1, 3)
    action2 = (3, 2, 1)

    discount_factor = 0.75
    annealing_factor = 1.0
    _, reward1, _, _ = sc.step(
        action1,
        discount_intermediate_reward=discount_factor,
        annealing_intermediate_reward=annealing_factor,
    )

    assert reward1 == SYNDROME_DIFF_REWARD * (
        0 + 2 * discount_factor + 2 * discount_factor ** 2 + 2 * discount_factor ** 3
    )

    _, reward2, _, _ = sc.step(
        action2,
        discount_intermediate_reward=discount_factor,
        annealing_intermediate_reward=annealing_factor,
    )

    assert reward2 == SYNDROME_DIFF_REWARD * (
        2 + 0 * discount_factor + 2 * discount_factor ** 2 - 2 * discount_factor ** 3
    )

    restore_env(original_depth, original_size, original_error_channel)
