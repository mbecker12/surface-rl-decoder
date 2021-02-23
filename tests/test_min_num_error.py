def test_min_x_errors(sc):
    err_channel = "x"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() < 5
        assert sc.actual_errors.sum() < 5

    sc.min_qbit_errors = 5
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() >= 5
        assert sc.actual_errors.sum() >= 5

def test_min_dp_errors(sc):
    err_channel = "dp"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() < 5
        assert sc.actual_errors.sum() < 5

    sc.min_qbit_errors = 5
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() >= 5
        assert sc.actual_errors.sum() >= 5

def test_min_iidxz_errors(sc):
    err_channel = "iidxz"
    sc.min_qbit_errors = 0
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() < 5
        assert sc.actual_errors.sum() < 5

    sc.min_qbit_errors = 5
    sc.p_error = 0.0001
    sc.p_msmt = 0.0001
    for _ in range(100):
        sc.reset(error_channel=err_channel)
        assert sc.qubits.sum() >= 5
        assert sc.actual_errors.sum() >= 5
