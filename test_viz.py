from surface_rl_decoder.surface_code import SurfaceCode

def test_qubit_grid():
    sc = SurfaceCode()
    sc.qubits[:, 2, 3] = 1
    sc.qubits[3:, 4, 1] = 1

    sc.qubits[:, 0, 0] = 3
    sc.qubits[:, 1, 3] = 3
    sc.qubits[4:, -1, 2] = 3

    sc.qubits[7:, 2, 0] = 2
    # sc.qubits[:, 1, 1] = 2
    sc.state = sc.create_syndrome_output_stack(sc.qubits)
    sc.render()


if __name__ == "__main__":
    test_qubit_grid()