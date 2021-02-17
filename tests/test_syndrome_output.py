import numpy as np
from surface_rl_decoder.surface_code import SurfaceCode


def test_syndrome_output(v=False):
    scode = SurfaceCode()

    scode.qubits[0, 1, 1] = 1

    expected_syndrome = np.zeros(
        (scode.system_size + 1, scode.system_size + 1), dtype=np.uint8
    )
    expected_syndrome[1, 1] = 1
    expected_syndrome[2, 2] = 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome)

    scode.qubits[0, 3, 2] = 1
    expected_syndrome[3, 3] += 1
    expected_syndrome[4, 2] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    # edge cases
    scode.qubits[0, 3, 4] = 1
    expected_syndrome[3, 5] += 1
    expected_syndrome[4, 4] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    scode.qubits[0, 2, 2] = 3  # z error
    expected_syndrome[2, 3] += 1
    expected_syndrome[3, 2] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    scode.qubits[0, 1, 3] = 2
    expected_syndrome[1, 3] += 1
    expected_syndrome[1, 4] += 1
    expected_syndrome[2, 3] += 1
    expected_syndrome[2, 4] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)


def test_syndrome_output_edge(v=True):
    scode = SurfaceCode()
    expected_syndrome = np.zeros(
        (scode.system_size + 1, scode.system_size + 1), dtype=np.uint8
    )

    scode.qubits[0, 0, 0] = 2

    expected_syndrome[0, 1] += 1
    expected_syndrome[1, 1] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    scode.qubits[0, 2, 0] = 2
    expected_syndrome[2, 0] += 1
    expected_syndrome[2, 1] += 1
    expected_syndrome[3, 1] += 1

    syndrome = scode.create_syndrome_output(scode.qubits[0])

    if v:
        print(f"{scode.qubits=}")
        print(f"{syndrome=}")
        print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)


# TODO
# def test_all_syndromes(v=True):
#     def reset():
#         scode = SurfaceCode()
#         expected_syndrome = np.zeros(
#             (scode.system_size + 1, scode.system_size + 1), dtype=np.uint8
#         )
#         return scode, expected_syndrome

#     scode = SurfaceCode()
#     d = scode.system_size

#     assert d == 5

#     # X errors
#     # 0,0
#     scode, expected_syndrome = reset()
#     scode.qubits[0, 0] = 1
#     expected_syndrome[]


#     # X errors
#     for i in range(d):
#         for j in range(d):
#             scode = SurfaceCode()
#             expected_syndrome = np.zeros(
#                 (scode.system_size + 1, scode.system_size + 1), dtype=np.uint8
#             )

#             scode.qubits[i, j] = 1
#             expected_syndrome[]


if __name__ == "__main__":
    # for debugging purposes
    test_syndrome_output()
