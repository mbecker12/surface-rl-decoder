"""
Create masks to denote where which type of syndrome is located.

The intention is to later use these masks to multiply to
a given syndrome calculation in order to aid vectorization.
"""
import numpy as np
from iniparser import Config
import torch

c = Config()
_config = c.scan(".", True).read()
config = c.config_rendered

env_config = config.get("config").get("env")
learner_config = config.get("config").get("learner")
device = learner_config.get("device")
d = int(env_config.get("size"))

# pylint: disable=pointless-string-statement
"""
Need to cover the following indices in a d=7 surface code to denote vertices
[
    [1, 1],
    [1, 3],
    [1, 5],
    [1, 7],

    [2, 0],
    [2, 2],
    [2, 4],
    [2, 6],

    [3, 1],
    [3, 3],
    [3, 5],
    [3, 7],

    [4, 0],
    [4, 2],
    [4, 4],
    [4, 6],

    [5, 1],
    [5, 3],
    [5, 5],
    [5, 7],

    [6, 0],
    [6, 2],
    [6, 4],
    [6, 6],
]
"""
vertex_mask = np.zeros((d + 1, d + 1), dtype=np.uint8)
for i in range(1, d):
    if i % 2 == 1:
        for j in range(1, d + 1, 2):
            vertex_mask[i, j] = 1
    else:
        for j in range(0, d + 1, 2):
            vertex_mask[i, j] = 1

vertex_mask_torch = torch.tensor(vertex_mask, device=device)
vertex_mask_torch_float = torch.tensor(vertex_mask, device=device, dtype=torch.float32)
vertex_mask_torch_int = torch.tensor(vertex_mask, device=device, dtype=torch.int64)

"""
Need to cover the following indices in a d=7 surface code to denote plaquettes
[
    [0, 1],
    [0, 3],
    [0, 5],

    [1, 2],
    [1, 4],
    [1, 6],

    [2, 1],
    [2, 3],
    [2, 5],

    [3, 2],
    [3, 4],
    [3, 6],

    [4, 1],
    [4, 3],
    [4, 5],

    [5, 2],
    [5, 4],
    [5, 6],

    [6, 1],
    [6, 3],
    [6, 5],

    [7, 2],
    [7, 4],
    [7, 6]
]
"""
plaquette_mask = np.zeros((d + 1, d + 1), dtype=np.uint8)
for i in range(1, d):
    if i % 2 == 1:
        for j in range(2, d + 1, 2):
            plaquette_mask[i, j] = 1
    else:
        for j in range(1, d, 2):
            plaquette_mask[i, j] = 1
for j in range(1, d - 1, 2):
    plaquette_mask[0, j] = 1
for j in range(2, d + 1, 2):
    plaquette_mask[d, j] = 1

plaquette_mask_torch = torch.tensor(plaquette_mask, device=device)
plaquette_mask_torch_float = torch.tensor(plaquette_mask, device=device, dtype=torch.float32)
plaquette_mask_torch_int = torch.tensor(plaquette_mask, device=device, dtype=torch.int64)