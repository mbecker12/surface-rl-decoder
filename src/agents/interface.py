"""
utility interface module, to split the input data into multiple
infromation channels
"""
from typing import Tuple
import torch
from torch import nn


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


def create_convolution_sequence(
    input_list, kernel_size, padding, convolution=nn.Conv2d, device="cpu"
):
    modules = []

    for i in range(len(input_list) - 1):
        modules.append(
            convolution(
                int(input_list[i]),
                int(input_list[i + 1]),
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        # modules.append(nn.ReLU())
    for layer in modules:
        layer.to(device)
    return modules
