import pytest
import numpy as np
from src.surface_rl_decoder.surface_code_util import TERMINAL_ACTION, perform_action


def test_step_function(sc):
    action = (1, 2, 3)
    sc.step(action)
    for i in range(sc.qubits.shape[0]):
        assert sc.qubits[i, 1, 2] == 3

    action = (0, 3, 4, 2)
    sc.step(action)
    for i in range(sc.qubits.shape[0]):
        assert sc.qubits[i, 1, 2] == 3
        assert sc.qubits[i, 3, 4] == 2

    action = (9, 3, 4, 3)
    sc.step(action)
    for i in range(sc.qubits.shape[0]):
        assert sc.qubits[i, 1, 2] == 3
        assert sc.qubits[i, 3, 4] == 1


def test_terminal_action(sc):
    action = (1, 2, TERMINAL_ACTION)
    sc.step(action)


def test_ground_state_with_errors(sc):
    """
    try to create a state which is not in the ground state
    and trigger the check_final_state function
    """
    dist = sc.system_size
    height = sc.stack_depth
    sc.qubits = np.zeros((height, dist, dist), dtype=np.uint8)

    actions = [(1, i, 3) for i in range(dist - 1)]

    for action in actions:
        sc.step(action)

    action = (0, 0, TERMINAL_ACTION)
    sc.step(action)

    assert not sc.ground_state, sc.qubits[-1]


def test_reset_function(sc):
    actions = [
        (1, 2, 3),
        (0, 4, 1),
        (2, 1, 2),
        (3, 3, 0),
    ]
    for action in actions:
        sc.step(action)

    sc.reset()


def test_util_perform_action(qbs):
    action = (0, 0, 2)
    perform_action(qbs, action)

    action = (0, 0, TERMINAL_ACTION)
    with pytest.raises(Exception):
        perform_action(qbs, action)


def test_non_ground_state_z(sc):
    """
    try to create a state which is not in the ground state
    and trigger the check_final_state function
    """
    dist = sc.system_size
    height = sc.stack_depth
    sc.qubits = np.zeros((height, dist, dist), dtype=np.uint8)

    actions = [(1, i, 3) for i in range(dist)]

    for action in actions:
        sc.step(action)

    action = (0, 0, TERMINAL_ACTION)
    sc.step(action)

    assert not sc.ground_state


def test_non_ground_state_x(sc):
    """
    try to create a state which is not in the ground state
    and trigger the check_final_state function
    """
    dist = sc.system_size
    height = sc.stack_depth
    sc.qubits = np.zeros((height, dist, dist), dtype=np.uint8)

    actions = [(i, 1, 1) for i in range(dist)]

    for action in actions:
        sc.step(action)

    action = (0, 0, TERMINAL_ACTION)
    sc.step(action)

    assert not sc.ground_state
