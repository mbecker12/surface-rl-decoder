"""
utility interface module, to split the input data into multiple
infromation channels
"""
from typing import Tuple
import torch


def interface(
    state: torch.Tensor, plaquette_mask: torch.Tensor, vertex_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    helper function to split up the input into three channels representing
    plaquette, vertex, and both syndrome types
    """
    x = state * plaquette_mask
    z = state * vertex_mask
    return x, z, state
