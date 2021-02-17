import pytest
import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode


def test_qubit_error_slice():
    scode = SurfaceCode()
    scode.p_error = 0.5
    for err_channel in ("dp", "x", "iidxz"):
        error_slice = scode.generate_qubit_error(error_channel=err_channel)

        assert error_slice.shape == (scode.system_size, scode.system_size)
        assert np.any(error_slice != 0)
    
    with pytest.raises(Exception):
        scode.generate_qubit_error(error_channel="nonsense")


def test_qubit_error_stack():
    scode = SurfaceCode()
    scode.p_error = 0.5
    for err_channel in ("dp", "x", "iidxz"):
        error_stack = scode.generate_qubit_error_stack(error_channel=err_channel)

        assert error_stack.shape == (
            scode.stack_depth,
            scode.system_size,
            scode.system_size,
        )
        assert np.any(error_stack != 0)

        # make sure that the number of errors in later layers
        # is geq number of errors in lower layers
        for height in range(scode.stack_depth - 1):
            assert scode.qubits[height].sum() <= scode.qubits[height + 1].sum()


def test_msmt_error_slice():
    scode = SurfaceCode()
    scode.p_msmt = 0.5

    erroneous_qubits = scode.generate_qubit_error()
    true_syndrome = scode.create_syndrome_output(erroneous_qubits)

    faulty_syndrome = scode.generate_measurement_error(true_syndrome)

    assert np.any(true_syndrome != faulty_syndrome)
    assert faulty_syndrome.shape == true_syndrome.shape
    assert faulty_syndrome.shape == (scode.syndrome_size, scode.syndrome_size)


def test_msmt_error_stack():
    scode = SurfaceCode()
    scode.p_msmt = 0.5

    erroneous_qubits = scode.generate_qubit_error_stack()
    true_syndrome = scode.create_syndrome_output_stack(erroneous_qubits)

    faulty_syndrome = scode.generate_measurement_error(true_syndrome)

    assert np.any(true_syndrome != faulty_syndrome)
    assert faulty_syndrome.shape == true_syndrome.shape
    assert faulty_syndrome.shape == (
        scode.stack_depth,
        scode.syndrome_size,
        scode.syndrome_size,
    )
