from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import create_syndrome_output_stack

def test_qubit_grid():
    sc = SurfaceCode()

    sc.syndrome_errors[0, 4, 1] = 1
    sc.syndrome_errors[3, 3, 3] = 1
    sc.syndrome_errors[4, 2, 4] = 1
    sc.syndrome_errors[6, 2, 2] = 1

    sc.qubits[:, 2, 3] = 1
    sc.qubits[3:, 4, 1] = 1

    sc.qubits[:, 0, 0] = 3
    sc.qubits[:, 1, 3] = 3
    sc.qubits[4:, -1, 2] = 3

    sc.qubits[7:, 2, 0] = 2

    sc.state = create_syndrome_output_stack(sc.qubits, sc.vertex_mask, sc.plaquette_mask)

    sc.render()

if __name__ == "__main__":
    test_qubit_grid()
