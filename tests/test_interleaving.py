import numpy as np

stack_depth = 8
system_size = 5

plaquette_size = (system_size - 1) // 2
vertex_size = (system_size + 1) // 2

plaquette_matrix = np.zeros((
    stack_depth,
    system_size + 1,
    plaquette_size
), dtype=np.int32)

vertex_matrix = np.zeros((
    stack_depth,
    system_size - 1,
    vertex_size
), dtype=np.int32)

# TODO: Test interleaving of plaquettes and vertices for history stack
# or come up with another smart way to represent this...
# this is especially difficult since vertex and plaquette matrices are not even of the same shape