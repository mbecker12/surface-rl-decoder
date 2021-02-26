import os

from src.surface_rl_decoder.surface_code import SurfaceCode

SUCCESS_RATE = 0.95
MIN_ERROR = 5
N_ITERATIONS = 100


def test_min_x_errors(sc):
    err_channel = "x"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() < MIN_ERROR and sc.actual_errors.sum() < MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE

    sc.min_qbit_errors = MIN_ERROR
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() >= MIN_ERROR and sc.actual_errors.sum() >= MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE


def test_min_dp_errors(sc):
    err_channel = "dp"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() < MIN_ERROR and sc.actual_errors.sum() < MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE

    sc.min_qbit_errors = MIN_ERROR
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() >= MIN_ERROR and sc.actual_errors.sum() >= MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE


def test_min_iidxz_errors(sc):
    err_channel = "iidxz"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() < MIN_ERROR and sc.actual_errors.sum() < MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE

    sc.min_qbit_errors = MIN_ERROR
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001

    successes = 0

    for _ in range(N_ITERATIONS):
        sc.reset(error_channel=err_channel)
        if sc.qubits.sum() >= MIN_ERROR and sc.actual_errors.sum() >= MIN_ERROR:
            successes += 1

    assert successes / N_ITERATIONS > SUCCESS_RATE


def test_init_nonzero_min_qbit_error():
    original_min_qb_error = os.environ.get("CONFIG_ENV_MIN_QBIT_ERR", "0")
    os.environ["CONFIG_ENV_MIN_QBIT_ERR"] = "3"

    successes = 0
    for _ in range(N_ITERATIONS):
        scode = SurfaceCode()
        scode.reset()
        if scode.qubits.sum() >= MIN_ERROR and scode.actual_errors.sum() >= MIN_ERROR:
            successes += 1

    os.environ["CONFIG_ENV_MIN_QBIT_ERR"] = original_min_qb_error
    assert successes / N_ITERATIONS > SUCCESS_RATE
