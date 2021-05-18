"""
Old implementation of the 3D Convolutional network.
It is needed for evaluation of trained networks which were trained using
this version of the implementation.
Implementation of an agent containing 3D convolutional layers
followed by linear layers
"""
from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from agents.interface import create_convolution_sequence, interface
from surface_rl_decoder.syndrome_masks import get_plaquette_mask, get_vertex_mask

NETWORK_SIZES = ["slim", "medium", "large", "extra_large"]


class Conv3dAgent(BaseAgent):

    """
    Description:
        Fourth iteration of an agent.
        Consists of multiple 3D convolutional layers.
        Splits the input into x,z and both errors which it
        then proceeds to feed each part into a neural network
        before adding them together and feeding them together
        into 2 linear layers from which only the final
        (with regards to time) output is used.
        For the instantiation it requires a dictionary
        containing the several parameters that the network
        will need to build itself.
    Parameters
    ==========
    config: dictionary containing configuration for the netwok. Expected keys:
        device: the device the code is working on
        split_input_toggle: an int indicating true or false of
            whether or not the input should be split
        code_size: the length of the surface code
        num_actions_per_cubit: in most cases should be 3 but can be
            adjusted, it is the number of types of correction actions
            that can be applied on the qubit
        stack_depth: the length of the dimension with aspect to time
        input_channels: the size of the number of channels at the input.
            Should be 1 in most cases but can be adjusted if one wishes
        output_channels: the number of output channels from the first 3d convolution
        output_channels2: the number of output channels from the second 3d convolution
        output_channels3: the number of output channels from the third 3d convolution
        output_channels4: the number of output channels from the fourth 3d convolution
        neurons_lin_layer: number of neurons for the second-to-last linear layer
        kernel_size: the size of the kernel
        padding_size: the amount of padding,
            should be a number such that the shortening in dimension length
            due to the kernel is negated
    """

    def __init__(self, config):
        super().__init__()

        self.device = config.get("device")
        # pylint: disable=not-callable
        self.size = int(config.get("code_size"))
        self.plaquette_mask = torch.tensor(
            get_plaquette_mask(self.size), device=self.device
        )
        self.vertex_mask = torch.tensor(get_vertex_mask(self.size), device=self.device)
        self.split_input_toggle = int(config.get("split_input_toggle", 1))
        self.input_channels = int(config.get("input_channels"))
        self.nr_actions_per_qubit = int(config.get("num_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))
        self.network_size = str(config.get("network_size"))
        assert self.network_size in NETWORK_SIZES

        self.kernel_size = int(config.get("kernel_size"))
        self.kernel_depth = int(config.get("kernel_depth", self.kernel_size))

        self.padding_size = int(config.get("padding_size"))
        if self.padding_size != 1:
            self.padding_size = (
                int(self.kernel_depth / 2),
                int(self.kernel_size / 2),
                int(self.kernel_size / 2),
            )

        self.kernel_size = (self.kernel_depth, self.kernel_size, self.kernel_size)
        input_channel_list: List = deepcopy(config.get("channel_list"))
        input_channel_list.insert(0, self.input_channels)
        layer_count = 0

        if self.network_size in NETWORK_SIZES:
            self.conv1 = nn.Conv3d(
                in_channels=input_channel_list[layer_count],
                out_channels=input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv2 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv3 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv4 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv5 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
        if self.network_size in NETWORK_SIZES[1:]:
            self.conv6 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv7 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv8 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
        if self.network_size in NETWORK_SIZES[2:]:
            self.conv9 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv10 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv11 = nn.Conv3d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1

        self.output_channels = int(input_channel_list[-1])
        self.neurons_output = self.nr_actions_per_qubit * self.size * self.size + 1
        self.cnn_dimension = (
            (self.size + 1) * (self.size + 1) * self.stack_depth * self.output_channels
        )

        input_neuron_numbers = config["neuron_list"]

        lin_layer_count = 0
        self.lin0 = nn.Linear(self.cnn_dimension, int(input_neuron_numbers[0]))
        if self.network_size in NETWORK_SIZES:
            self.lin1 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )
            lin_layer_count += 1
        if self.network_size in NETWORK_SIZES[1:]:
            self.lin2 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )
            lin_layer_count += 1

        self.output_layer = nn.Linear(
            int(input_neuron_numbers[-1]), self.neurons_output
        )

    def forward(self, state: torch.Tensor):
        """
        forward pass
        """
        # state = self._format(state, device=self.device)
        batch_size = state.shape[0]

        both = state.view(
            -1,
            self.input_channels,
            self.stack_depth,
            (self.size + 1),
            (self.size + 1),
        )  # convolve both

        if self.network_size in NETWORK_SIZES:
            both = F.relu(self.conv1(both))
            both = F.relu(self.conv2(both))
            both = F.relu(self.conv3(both))
            both = F.relu(self.conv4(both))
            both = F.relu(self.conv5(both))

        if self.network_size in NETWORK_SIZES[1:]:
            both = F.relu(self.conv6(both))
            both = F.relu(self.conv7(both))
            both = F.relu(self.conv8(both))
        if self.network_size in NETWORK_SIZES[2:]:
            both = F.relu(self.conv9(both))
            both = F.relu(self.conv10(both))
            both = F.relu(self.conv11(both))

        complete = both.view(
            -1,
            (self.size + 1) * (self.size + 1) * self.stack_depth * self.output_channels,
        )  # make sure the dimensions are in order
        complete = F.relu(self.lin0(complete))
        if self.network_size in NETWORK_SIZES:
            complete = F.relu(self.lin1(complete))
        if self.network_size in NETWORK_SIZES[1:]:
            complete = F.relu(self.lin2(complete))

        final_output = self.output_layer(complete)
        return final_output