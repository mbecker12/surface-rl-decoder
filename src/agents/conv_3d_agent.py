"""
Implementation of an agent containing 3D convolutional layers
followed by linear layers
"""
from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent
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
        self.rl_type = str(config.get("rl_type", "q"))
        assert self.rl_type in ("q", "ppo")

        # self.activation_fn = F.relu
        self.activation_fn = F.silu

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

            if self.rl_type == "ppo":
                ppo_input_channels = input_channel_list[layer_count]
                ppo_output_channels = input_channel_list[layer_count + 1]
                self.conv_val1 = nn.Conv3d(
                    ppo_input_channels,
                    ppo_output_channels,
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

            if self.rl_type == "ppo":
                ppo_input_channels = ppo_output_channels
                ppo_output_channels = input_channel_list[layer_count + 1]
                self.conv_val2 = nn.Conv3d(
                    ppo_input_channels,
                    ppo_output_channels,
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
            if self.rl_type == "ppo":
                ppo_input_channels = ppo_output_channels
                ppo_output_channels = input_channel_list[layer_count + 1]
                self.conv_val3 = nn.Conv3d(
                    ppo_input_channels,
                    ppo_output_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding_size,

                )
            layer_count += 1

        self.output_channels = int(input_channel_list[-1])
        self.neurons_output = self.nr_actions_per_qubit * self.size * self.size + 1

        self.cnn_dimension = (
            (self.size + 1) * (self.size + 1) * self.stack_depth * self.output_channels
        )
        if self.rl_type == "ppo":
            self.cnn_val_dimension = (
                (self.size + 1)
                * (self.size + 1)
                * self.stack_depth
                * ppo_output_channels
            )

        input_neuron_numbers = config["neuron_list"]

        lin_layer_count = 0
        self.lin0 = nn.Linear(self.cnn_dimension, int(input_neuron_numbers[0]))

        if self.rl_type == "ppo":
            self.lin_val0 = nn.Linear(
                self.cnn_val_dimension, int(input_neuron_numbers[0])
            )
        if self.network_size in NETWORK_SIZES:
            self.lin1 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )

            if self.rl_type == "ppo":
                self.lin_val1 = nn.Linear(
                    input_neuron_numbers[lin_layer_count],
                    input_neuron_numbers[lin_layer_count + 1],
                )

            lin_layer_count += 1

        if self.network_size in NETWORK_SIZES[1:]:
            self.lin2 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )

            if self.rl_type == "ppo":
                self.lin_val2 = nn.Linear(
                    input_neuron_numbers[lin_layer_count],
                    input_neuron_numbers[lin_layer_count + 1],
                )
            lin_layer_count += 1

        self.output_layer = nn.Linear(
            int(input_neuron_numbers[-1]), self.neurons_output
        )
        if self.rl_type == "ppo":
            self.output_value_layer = nn.Linear(int(input_neuron_numbers[-1]), 1)

        # for param_tensor in self.state_dict():
        #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def forward(self, state: torch.Tensor):
        """
        forward pass
        """
        state = self._format(state)
        batch_size = state.shape[0]

        both0 = state.view(
            -1,
            self.input_channels,
            self.stack_depth,
            (self.size + 1),
            (self.size + 1),
        )  # convolve both

        # convolutions
        if self.network_size in NETWORK_SIZES:
            both1 = self.activation_fn(self.conv1(both0))
            both2 = self.activation_fn(self.conv2(both1))
            both3 = self.activation_fn(self.conv3(both2))
            both4 = self.activation_fn(self.conv4(both3))
            if self.rl_type == "ppo":
                values1 = self.activation_fn(self.conv_val1(both4))
            both5 = self.activation_fn(self.conv5(both4))

        if self.network_size in NETWORK_SIZES[1:]:
            both6 = self.activation_fn(self.conv6(both5))
            both7 = self.activation_fn(self.conv7(both6))
            both8 = self.activation_fn(self.conv8(both7))
            if self.rl_type == "ppo":
                values2 = self.activation_fn(self.conv_val2(values1))

        if self.network_size in NETWORK_SIZES[2:]:
            both9 = self.activation_fn(self.conv9(both8))
            both10 = self.activation_fn(self.conv10(both9))
            both11 = self.activation_fn(self.conv11(both10))
            if self.rl_type == "ppo":
                values3 = self.activation_fn(self.conv_val3(values2))

        # reshape, dependent on network size
        if self.network_size in NETWORK_SIZES[2:]:
            complete0 = both11.view(-1, self.cnn_dimension)
            if self.rl_type == "ppo":
                values_complete0 = values3.view(-1, self.cnn_val_dimension)
        elif self.network_size in NETWORK_SIZES[1]:
            complete0 = both8.view(-1, self.cnn_dimension)
            if self.rl_type == "ppo":
                values_complete0 = values2.view(-1, self.cnn_val_dimension)
        elif self.network_size in NETWORK_SIZES[0]:
            complete0 = both5.view(-1, self.cnn_dimension)
            if self.rl_type == "ppo":
                values_complete0 = values1.view(-1, self.cnn_val_dimension)

        else:
            raise Exception(f"Network size {self.network_size} not supported.")

        # first linear layer
        if self.rl_type == "ppo":
            values_complete1 = self.activation_fn(self.lin_val0(values_complete0))
        complete1 = self.activation_fn(self.lin0(complete0))

        # remaining linear layers
        if self.network_size in NETWORK_SIZES:
            complete2 = self.activation_fn(self.lin1(complete1))
            if self.rl_type == "ppo":
                values_complete2 = self.activation_fn(self.lin_val1(values_complete1))
        if self.network_size in NETWORK_SIZES[1:]:
            complete3 = self.activation_fn(self.lin2(complete2))
            if self.rl_type == "ppo":
                values_complete3 = self.activation_fn(self.lin_val2(values_complete2))

        # final output
        if self.network_size in NETWORK_SIZES[1:]:
            final_output = self.output_layer(complete3)
            if self.rl_type == "ppo":
                final_values = self.output_value_layer(values_complete3)
                return final_output, final_values
            return final_output

        elif self.network_size in NETWORK_SIZES[0]:
            final_output = self.output_layer(complete2)
            if self.rl_type == "ppo":
                final_values = self.output_value_layer(values_complete2)
                return final_output, final_values
            return final_output
        else:
            raise Exception(f"Network size {self.network_size} not supported.")
