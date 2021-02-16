import pytest
import numpy as np
from surface_code import SurfaceCode


def test_syndrome_output(v=False):
    sc = SurfaceCode()

    sc.qubits[1, 1] = 1
    
    expected_syndrome = np.zeros((sc.system_size + 1, sc.system_size + 1), dtype=np.uint8)
    expected_syndrome[1, 1] = 1
    expected_syndrome[2, 2] = 1
    
    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome)

    sc.qubits[3, 2] = 1
    expected_syndrome[3, 3] += 1
    expected_syndrome[4, 2] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    # edge cases
    sc.qubits[3, 4] = 1
    expected_syndrome[3, 5] += 1
    expected_syndrome[4, 4] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[2, 2] = 3 # z error
    expected_syndrome[2, 3] += 1
    expected_syndrome[3, 2] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[1, 3] = 2
    expected_syndrome[1, 3] += 1
    expected_syndrome[1, 4] += 1
    expected_syndrome[2, 3] += 1
    expected_syndrome[2, 4] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

def test_syndrome_output_edge(v=True):
    sc = SurfaceCode()
    expected_syndrome = np.zeros((sc.system_size + 1, sc.system_size + 1), dtype=np.uint8)
    
    sc.qubits[0, 0] = 2
    
    expected_syndrome[0, 1] += 1
    expected_syndrome[1, 1] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits[2, 0] = 2
    expected_syndrome[2, 0] += 1
    expected_syndrome[2, 1] += 1
    expected_syndrome[3, 1] += 1

    syndrome = sc.create_syndrome_output(sc.qubits)

    if v: print(f"{sc.qubits=}")
    if v: print(f"{syndrome=}")
    if v: print(f"{expected_syndrome=}")
    assert np.all(syndrome == expected_syndrome % 2)

    sc.qubits

# TODO
# def test_all_syndromes(v=True):
#     def reset():
#         sc = SurfaceCode()
#         expected_syndrome = np.zeros((sc.system_size + 1, sc.system_size + 1), dtype=np.uint8)
#         return sc, expected_syndrome

#     sc = SurfaceCode()
#     d = sc.system_size

#     assert d == 5

#     # X errors
#     # 0,0
#     sc, expected_syndrome = reset()
#     sc.qubits[0, 0] = 1
#     expected_syndrome[]


#     # X errors
#     for i in range(d):
#         for j in range(d):
#             sc = SurfaceCode()
#             expected_syndrome = np.zeros((sc.system_size + 1, sc.system_size + 1), dtype=np.uint8)

#             sc.qubits[i, j] = 1
#             expected_syndrome[]
            

if __name__ == "__main__":
    # for debugging purposes
    test_syndrome_output()