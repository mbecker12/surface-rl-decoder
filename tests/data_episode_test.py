"""
Data generated by

    sc = SurfaceCode()
    np.random.seed(42)
    sc.p_error = 0.1
    sc.p_msmt = 0.1
    sc.reset()

for d = 5
    h = 4
"""
import numpy as np

_qubits = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 3, 1, 0, 0],
            [2, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 3, 1, 3, 0],
            [2, 0, 2, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [3, 3, 1, 3, 0],
            [2, 0, 2, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    ],
    dtype=np.uint8,
)


_syndrome_errors = np.array(
    [
        [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, False],
        ],
        [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
        [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
        [
            [False, False, False, False, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
    ],
    dtype=bool,
)


_actual_errors = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 3, 1, 0, 0],
            [2, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 3, 1, 3, 0],
            [2, 0, 2, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [3, 3, 1, 3, 0],
            [2, 0, 2, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    ],
    dtype=np.uint8,
)

_actions = [
    (0, 2, 1),
    (1, 0, 3),
    (1, 1, 3),
    (1, 2, 1),
    (1, 3, 3),
    (2, 0, 2),
    (2, 2, 2),
    (3, 2, 2),
    (4, 3, 1),
]

_state = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
    ],
    dtype=np.uint8,
)
