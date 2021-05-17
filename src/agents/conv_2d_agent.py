"""
Implementation of an agent containing convolutional layers
followed by an LSTM to account for the time dependency
and linear layers to generate q values.
"""
from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtrxl_torch.gtrxl_torch import GTrXL
from agents.base_agent import BaseAgent
from surface_rl_decoder.syndrome_masks import get_vertex_mask, get_plaquette_mask
from agents.interface import create_convolution_sequence, interface

NETWORK_SIZES = ["slim", "medium", "large", "extra_large"]


class Conv2dAgent(BaseAgent):

    """
    Description:
        Third iteration of an agent. Consists of multiple 2D convolutional layers and an LSTM layer.
        Splits the input into x, z and both errors which it then proceeds to feed each part into a
        neural network before convolving them to a single feature map and adding them together,
        feeding them into the LSTM layer and finally into 2 linear layers
        from which only the final (with regards to time) output is used.
        For the instantiation it requires a dictionary containing the several
        parameters that the network will need to build itself.

    Parameters
    ==========
    config: dictionary containing configuration for the network. Expected keys:
        code_size: the code distance of the surface code
        num_actions_per_qubit: in most cases should be 3 but can be adjusted,
            it is the number of types of correction actions that can be applied on the qubit
        stack_depth: the length of the dimension with aspect to time
        input_channels: the size of the number of channels at the input.
            Should be 1 in most cases but can be adjusted if one wishes
        output_channels: the number of output channels from the first 2d convolution
        output_channels2: the number of output channels from the second 2d convolution
        output_channels3: the number of output channels from the third 2d convolution
        lstm_num_layers: the lstm network has a parameter which queries for
            how many lstm layers should be stacked on top of each other
        lstm_num_directions: can be 1 or 2, for uni- or bidirectional LSTM
        lstm_output_size: number of features in the hidden state of the LSTM;
            used synonymously as the size of the LSTM output vector
        neurons_lin_layer: number of neurons in the second-to-last linear layer

        kernel_size: the size of the convolution kernel
        padding_size: the amount of padding, should be a number
            such that the shortening in dimension length due to the kernel is negated
    """

    def __init__(self, config):
        super().__init__()
        self.device = config.get("device")
        assert self.device is not None
        self.size = int(config.get("code_size"))
        # pylint: disable=not-callable
        self.plaquette_mask = torch.tensor(
            get_plaquette_mask(self.size), device=self.device
        )
        self.vertex_mask = torch.tensor(get_vertex_mask(self.size), device=self.device)

        self.nr_actions_per_qubit = int(config.get("num_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))
        self.split_input_toggle = int(config.get("split_input_toggle", 1))

        self.input_channels = int(config.get("input_channels"))
        self.kernel_size = int(config.get("kernel_size"))
        self.padding_size = int(config.get("padding_size"))
        self.network_size = str(config.get("network_size"))
        assert self.network_size in NETWORK_SIZES

        self.use_lstm = int(config.get("use_lstm", 0))
        self.use_rnn = int(config.get("use_rnn", 0))
        self.use_transformer = int(config.get("use_gtrxl", 0))
        self.use_transformer += int(config.get("use_transformer", 0))
        self.use_gru = int(config.get("use_gru", 0))
        self.use_all_rnn_layers = int(config.get("use_all_rnn_layers", 0))

        if self.use_transformer:
            self.gtrxl_heads = int(config.get("gtrxl_heads"))
            self.gtrxl_layers = int(config.get("gtrxl_layers"))
            self.gtrxl_hidden_dims = int(config.get("gtrxl_hidden_dims", 2048))
            self.gtrxl_rnn_layers = int(config.get("gtrxl_rnn_layers", 1))

        if self.use_lstm or self.use_rnn:
            self.lstm_num_layers = int(config.get("lstm_num_layers"))
            self.lstm_num_directions = int(config.get("lstm_num_directions"))
            assert self.lstm_num_directions in (1, 2)
            self.lstm_is_bidirectional = bool(self.lstm_num_directions - 1)
            self.lstm_output_size = int(config.get("lstm_output_size"))

        input_channel_list: List = deepcopy(config.get("channel_list"))
        input_channel_list.insert(0, self.input_channels)
        layer_count = 0
        if self.network_size in NETWORK_SIZES:
            self.conv1 = nn.Conv2d(
                in_channels=input_channel_list[layer_count],
                out_channels=input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv2 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.norm1 = nn.BatchNorm2d(input_channel_list[layer_count])
            self.conv3 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv4 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.norm2 = nn.BatchNorm2d(input_channel_list[layer_count])

        if self.network_size in NETWORK_SIZES[1:]:
            self.conv5 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv6 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv7 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.norm3 = nn.BatchNorm2d(input_channel_list[layer_count])

        if self.network_size in NETWORK_SIZES[2:]:
            self.conv8 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.conv9 = nn.Conv2d(
                input_channel_list[layer_count],
                input_channel_list[layer_count + 1],
                kernel_size=self.kernel_size,
                padding=self.padding_size,
            )
            layer_count += 1
            self.norm4 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.output_channels = input_channel_list[-1]

        self.neurons_output = self.nr_actions_per_qubit * self.size * self.size + 1
        self.cnn_dimension = (self.size + 1) * (self.size + 1) * self.output_channels

        linear_modules = []
        input_neuron_numbers = config["neuron_list"]

        if self.use_lstm:
            print("Using LSTM")
            self.lstm_layer = nn.LSTM(
                self.cnn_dimension,
                self.lstm_output_size,
                num_layers=self.lstm_num_layers,
                bidirectional=self.lstm_is_bidirectional,
                batch_first=True,
            )
            lstm_total_output_size = self.lstm_output_size * self.lstm_num_directions
            if self.use_all_rnn_layers:
                lstm_total_output_size *= self.stack_depth

            lin_layer_count = 0
            self.lin0 = nn.Linear(lstm_total_output_size, int(input_neuron_numbers[0]))

        elif self.use_transformer:
            print("Using GTRXL")
            self.gtrxl_dimension = self.cnn_dimension
            self.gated_transformer = GTrXL(
                d_model=self.gtrxl_dimension,
                nheads=self.gtrxl_heads,
                transformer_layers=self.gtrxl_layers,
                hidden_dims=self.gtrxl_hidden_dims,
                n_layers=self.gtrxl_rnn_layers,
                activation="gelu",
            )

            gtrxl_total_output_size = self.gtrxl_dimension
            if self.use_all_rnn_layers:
                gtrxl_total_output_size *= self.stack_depth

            lin_layer_count = 0
            self.lin0 = nn.Linear(gtrxl_total_output_size, int(input_neuron_numbers[0]))

        elif self.use_gru:
            print("GRU not supported yet")

        else:
            print("Not using any recurrent module")
            lin_layer_count = 0
            self.lin0 = nn.Linear(
                self.cnn_dimension * self.stack_depth, int(input_neuron_numbers[0])
            )

        if self.network_size in NETWORK_SIZES:
            self.lin1 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )
            lin_layer_count += 1
        if self.network_size in NETWORK_SIZES[2:]:
            self.lin2 = nn.Linear(
                input_neuron_numbers[lin_layer_count],
                input_neuron_numbers[lin_layer_count + 1],
            )
            lin_layer_count += 1

        self.output_layer = nn.Linear(
            input_neuron_numbers[-1], int(self.neurons_output)
        )

    def forward(self, state: torch.Tensor):
        """
        Perform a forward pass with the current batch of syndrome states.
        """
        # multiple input channels for different procedures,
        # they are then concatenated as the data is processed
        state = self._format(state)
        batch_size, _, _, _ = state.size()

        both = state.view(
            -1, self.input_channels, (self.size + 1), (self.size + 1)
        )  # convolve both
        if self.network_size in NETWORK_SIZES:
            both = F.silu(self.conv1(both))
            both = self.conv2(both)
            both = F.silu(self.norm1(both))
            both = F.silu(self.conv3(both))
            both = self.conv4(both)
            both = F.silu(self.norm2(both))

        if self.network_size in NETWORK_SIZES[1:]:
            both = F.silu(self.conv5(both))
            both = F.silu(self.conv6(both))
            both = self.conv7(both)
            both = F.silu(self.norm3(both))

        if self.network_size in NETWORK_SIZES[2:]:
            both = F.silu(self.conv8(both))
            both = self.conv9(both)
            both = F.silu(self.norm4(both))

        # convert the data back to <batch_size> samples of syndrome volumes
        # with <stack_depth> layers
        complete = both.view(batch_size, self.stack_depth, self.cnn_dimension)

        if self.use_lstm:
            output, (_h, _c) = self.lstm_layer(complete)

            if not self.use_all_rnn_layers:
                output = output[:, -1, :]

        elif self.use_transformer:
            complete = complete.permute(1, 0, 2)
            output = self.gated_transformer(complete)

            if not self.use_all_rnn_layers:
                output = output[-1, :, :]
            else:
                output = output.permute(1, 0, 2)

        else:
            output = complete.view(batch_size, -1)

        output = output.view(batch_size, -1)
        complete = F.silu(self.lin0(output))
        if self.network_size in NETWORK_SIZES:
            complete = F.silu(self.lin1(complete))
        if self.network_size in NETWORK_SIZES[2:]:
            complete = F.silu(self.lin2(complete))

        final_output = self.output_layer(complete)
        return final_output
