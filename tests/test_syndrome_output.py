import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode
from src.surface_rl_decoder.surface_code_util import create_syndrome_output


def test_syndrome_output(v=False):
    sc = SurfaceCode()

    sc.qubits[0, 1, 1] = 1

    expected_syndrome = np.zeros(
        (sc.system_size + 1, sc.system_size + 1), dtype=np.uint8
    )
    expected_syndrome[1, 2] = 1
    expected_syndrome[2, 1] = 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome)

    sc.qubits[0, 3, 2] = 1
    expected_syndrome[3, 2] += 1
    expected_syndrome[4, 3] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    # edge cases
    sc.qubits[0, 3, 4] = 1
    expected_syndrome[3, 4] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[0, 2, 2] = 3  # z error
    expected_syndrome[2, 2] += 1
    expected_syndrome[3, 3] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[0, 1, 3] = 2  # y error
    expected_syndrome[1, 3] += 1
    expected_syndrome[1, 4] += 1
    expected_syndrome[2, 3] += 1
    expected_syndrome[2, 4] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)


def test_syndrome_output_edge(v=True):
    sc = SurfaceCode()
    expected_syndrome = np.zeros(
        (sc.system_size + 1, sc.system_size + 1), dtype=np.uint8
    )

    sc.qubits[0, 0, 0] = 2

    expected_syndrome[0, 1] += 1
    expected_syndrome[1, 1] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[0, 2, 0] = 2
    expected_syndrome[2, 0] += 1
    expected_syndrome[2, 1] += 1
    expected_syndrome[3, 1] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    if v:
        print(f"{sc.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)


def test_multiple_syndromes_x(sc, configure_env, restore_env):
    expected_syndrome = np.zeros(
        (sc.system_size + 1, sc.system_size + 1), dtype=np.uint8
    )

    original_depth, original_size, original_error_channel = configure_env()
    sc = SurfaceCode()
    d = sc.system_size

    # need to restrict system size here to be able to check boundaries
    assert d == 5

    # X errors
    # 0,0
    sc.qubits[0, 0, 0] = 1
    expected_syndrome[0, 1] += 1

    # 0,2
    sc.qubits[0, 0, 2] = 1
    expected_syndrome[1, 2] += 1
    expected_syndrome[0, 3] += 1

    # 1,4
    sc.qubits[0, 1, 4] = 1
    expected_syndrome[1, 4] += 1

    # 2,2
    sc.qubits[0, 2, 2] = 1
    expected_syndrome[2, 3] += 1
    expected_syndrome[3, 2] += 1

    # 4,0
    sc.qubits[0, 4, 1] = 1
    expected_syndrome[4, 1] += 1
    expected_syndrome[5, 2] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)
    assert np.all(syndrome == expected_syndrome % 2)

    restore_env(original_depth, original_size, original_error_channel)


def test_multiple_syndromes_z(sc, configure_env, restore_env):
    expected_syndrome = np.zeros(
        (sc.system_size + 1, sc.system_size + 1), dtype=np.uint8
    )

    original_depth, original_size, original_error_channel = configure_env()
    sc = SurfaceCode()
    d = sc.system_size

    # need to restrict system size here to be able to check boundaries
    assert d == 5

    # Z errors
    # 4,1
    sc.qubits[0, 4, 1] = 3
    expected_syndrome[4, 2] += 1

    # 2,2
    sc.qubits[0, 2, 2] = 3
    expected_syndrome[2, 2] += 1
    expected_syndrome[3, 3] += 1

    # 1,4
    sc.qubits[0, 1, 3] = 3
    expected_syndrome[1, 3] += 1
    expected_syndrome[2, 4] += 1

    # 4,4
    sc.qubits[0, 3, 4] = 3
    expected_syndrome[4, 4] += 1
    expected_syndrome[3, 5] += 1

    syndrome = create_syndrome_output(sc.qubits[0], sc.vertex_mask, sc.plaquette_mask)

    assert np.all(syndrome == expected_syndrome % 2)

    restore_env(original_depth, original_size, original_error_channel)


if __name__ == "__main__":
    # for debugging purposes
    test_syndrome_output()
