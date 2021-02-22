from copy import deepcopy
import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode
from src.surface_rl_decoder.surface_code_util import TERMINAL_ACTION
from tests.data_episode_test import (
    _actual_errors,
    _qubits,
    _state,
    _syndrome_errors,
    _actions,
)


def test_episode(sc):
    sc.p_error = 0.0
    sc.p_msmt = 0.0
    sc.reset()

    assert sc.qubits.sum() == 0
    assert sc.state.sum() == 0
    assert sc.next_state.sum() == 0

    actions = [(0, 0, 0), (1, 3, 2), (4, 0, 1), (1, 1, 3), (3, 2, 1)]
    h_max = sc.stack_depth - len(actions) + 1

    # prepare qubit grid
    for i, action in enumerate(actions):
        row, col, operator = action[:]
        sc.qubits[i:, row, col] = operator

    assert sc.qubits.sum() != 0

    sc.actual_errors = deepcopy(sc.qubits)
    assert sc.actual_errors.sum() != 0

    # assert that errors have been applied
    for i in range(sc.qubits.shape[0] - 1):
        assert sc.qubits[i + 1].sum() >= sc.qubits[i].sum(), np.vstack(
            (sc.qubits[i], sc.qubits[i + 1])
        )

    # apply actions to resolve qubit errors
    for action in actions:
        previous_qb_sum = sc.qubits.sum()
        sc.step(action)
        assert previous_qb_sum >= sc.qubits.sum()

    # send terminal action to stop the episode
    sc.step(action=(9, 9, TERMINAL_ACTION))

    # assure that the action history corresponds to the created actions
    for i, action in enumerate(actions):
        assert np.all(sc.actions[i] == action)

    # in the bottommost layers, new pseudo-errors will have been created
    # make sure that those do exist in the final stack
    for h in range(sc.stack_depth - len(actions)):
        qb_sum = sc.qubits[h].sum()

        assert qb_sum != 0, h

    # in the topmost layers however, all errors should have been resolved
    assert sc.qubits[h_max:].sum() == 0, sc.qubits

    sc.step(action=(9, 9, TERMINAL_ACTION))
    assert sc.qubits[h_max:].sum() == 0, sc.qubits


def test_episode_w_measurement_errors(sc, block=False):
    sc.p_error = 0.0
    sc.p_msmt = 0.0
    sc.reset()

    assert sc.qubits.sum() == 0
    assert sc.state.sum() == 0
    assert sc.next_state.sum() == 0

    actions = [(0, 0, 0), (1, 2, 3), (4, 3, 2), (0, 1, 1), (3, 0, 2)]

    h_max = sc.stack_depth - len(actions) + 1

    # prepare qubit grid
    for i, action in enumerate(actions):
        row, col, operator = action[:]
        sc.qubits[i:, row, col] = operator

    assert sc.qubits.sum() != 0

    sc.actual_errors = deepcopy(sc.qubits)
    assert sc.actual_errors.sum() != 0

    sc.render(block=block)

    # introduce measurement errors
    sc.syndrome_errors = deepcopy(sc.state)
    sc.syndrome_errors[1, 3, 5] = 1
    sc.syndrome_errors[2, 3, 4] = 1
    sc.syndrome_errors[4, 1, 2] = 1
    sc.syndrome_errors[:2, 4, 1] = 1
    sc.syndrome_errors[6:, 4, 1] = 1

    # apply actions to resolve qubit errors
    for action in actions:
        previous_qb_sum = sc.qubits.sum()
        sc.step(action)
        assert previous_qb_sum >= sc.qubits.sum()

    assert sc.state[1, 1, 1] == 1
    assert sc.state[1, 4, 0] == 1
    assert sc.state[1, 3, 1] == 1
    assert sc.state[1, 4, 1] == 0
    assert sc.state[1, 4, 3] == 1
    assert sc.state[1, 4, 4] == 1
    assert sc.state[1, 3, 5] == 1
    assert sc.syndrome_errors[1, 3, 5] == 1
    assert sc.actual_errors[1, 3, 4] == 0
    assert sc.actual_errors[1, 2, 4] == 0

    assert sc.state[2, 4, 1] == 1
    assert sc.state[2, 1, 1] == 1
    assert sc.state[2, 3, 4] == 1

    assert sc.state[4, 1, 2] == 1
    assert sc.state[6, 4, 1] == 1

    assert sc.qubits[h_max:].sum() == 0, sc.qubits

    sc.step(action=(9, 9, TERMINAL_ACTION))
    assert sc.qubits[h_max:].sum() == 0, sc.qubits

    sc.render(block=block)


def test_proper_episode(configure_env, restore_env, seed_surface_code):
    """
    Test a proper episode where no human inference occurs
    which could corrupt and disturb copy actions etc.
    """

    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    seed_surface_code(sc, 42, 0.1, 0.1, "dp")

    assert sc.stack_depth == 4
    assert sc.system_size == 5

    # assure that arrays are the same after seeding the rng
    assert np.all(sc.qubits == _qubits), sc.qubits
    assert np.all(sc.syndrome_errors == _syndrome_errors), sc.syndrome_errors
    assert np.all(sc.actual_errors == _actual_errors), sc.actual_errors
    assert np.all(sc.state == _state), sc.state

    restore_env(original_depth, original_size, original_error_channel)

    for action in _actions:
        sc.step(action)

    for i, action in enumerate(_actions):
        assert np.all(sc.actions[i] == action)

    terminal_action = np.asarray((999, 999999, TERMINAL_ACTION), dtype=sc.actions.dtype)
    sc.step(terminal_action)

    assert np.all(sc.actions[len(_actions)] == terminal_action), sc.actions[:10]

    assert sc.qubits[0, 1, 1] == 0
    assert sc.qubits[0, 2, 0] == 0
    assert sc.qubits[0, 0, 2] == 1
    assert sc.qubits[0, 3, 2] == 2
    assert sc.qubits[0, 1, 3] == 3
    assert sc.qubits[-1].sum() == 0
    assert sc.state[3, 1, 4] != 0
    assert sc.state[3, 3, 3] != 0

    # the base error arrays should not change by taking actions
    assert np.all(sc.actual_errors == _actual_errors), sc.actual_errors
    assert np.all(sc.syndrome_errors == _syndrome_errors), sc.syndrome_errors

    # make sure that after the actions, only measurement errors are left in the
    # topmost layer
    assert np.all(sc.state[-1] == _syndrome_errors[-1]), _syndrome_errors[-1]
    # ...and if we take away the measurement errors, no errors remain
    assert np.all(np.logical_xor(sc.state[-1], _syndrome_errors[-1]) == 0)

    assert sc.ground_state


if __name__ == "__main__":
    scode = SurfaceCode()
    test_episode_w_measurement_errors(scode, block=True)
