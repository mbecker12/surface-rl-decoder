from copy import deepcopy
import numpy as np
from src.surface_rl_decoder.surface_code_util import TERMINAL_ACTION


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

    for i in range(sc.qubits.shape[0] - 1):
        assert sc.qubits[i + 1].sum() >= sc.qubits[i].sum(), np.vstack(
            (sc.qubits[i], sc.qubits[i + 1])
        )

    # apply actions to resolve qubit errors
    for action in actions:
        previous_qb_sum = sc.qubits.sum()
        sc.step(action)

        assert previous_qb_sum >= sc.qubits.sum()

    sc.step(action=(9, 9, TERMINAL_ACTION))

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
