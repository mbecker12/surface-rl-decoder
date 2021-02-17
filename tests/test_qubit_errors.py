import pytest
import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode


def test_qubit_error_slice():
    sc = SurfaceCode()
    sc.p_error = 0.5
    for err_channel in ("dp", "x", "iidxz"):
        error_slice = sc.generate_qubit_error(error_channel=err_channel)

        assert error_slice.shape == (sc.system_size, sc.system_size)
        assert np.any(error_slice != 0)


def test_qubit_error_stack():
    sc = SurfaceCode()
    sc.p_error = 0.5
    for err_channel in ("dp", "x", "iidxz"):
        error_stack = sc.generate_qubit_error_stack(error_channel=err_channel)

        assert error_stack.shape == (sc.stack_depth, sc.system_size, sc.system_size)
        assert np.any(error_stack != 0)

        # TODO: make sure that the number of errors in later layers is geq number of errors in lower layers


def test_msmt_error_slice():
    sc = SurfaceCode()
    sc.p_msmt = 0.5

    erroneous_qubits = sc.generate_qubit_error()
    true_syndrome = sc.create_syndrome_output(erroneous_qubits)

    faulty_syndrome = sc.generate_measurement_error(true_syndrome)

    assert np.any(true_syndrome != faulty_syndrome)
    assert faulty_syndrome.shape == true_syndrome.shape
    assert faulty_syndrome.shape == (sc.syndrome_size, sc.syndrome_size)


def test_msmt_error_stack():
    sc = SurfaceCode()
    sc.p_msmt = 0.5

    erroneous_qubits = sc.generate_qubit_error_stack()
    true_syndrome = sc.create_syndrome_output_stack(erroneous_qubits)

    faulty_syndrome = sc.generate_measurement_error(true_syndrome)

    assert np.any(true_syndrome != faulty_syndrome)
    assert faulty_syndrome.shape == true_syndrome.shape
    assert faulty_syndrome.shape == (sc.stack_depth, sc.syndrome_size, sc.syndrome_size)
