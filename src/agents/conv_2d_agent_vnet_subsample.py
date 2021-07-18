from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from surface_rl_decoder.syndrome_masks import get_vertex_mask, get_plaquette_mask

import logging

logger = logging.getLogger(name="MODEL")


class Conv2dAgentValueNetSubsample(BaseAgent):
    def __init__(self, config):
        super().__init__()
        # logger.info("Conv2dAgentValueNetSubsample")
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

        # self.input_channels = int(config.get("input_channels"))
        self.input_channels = self.stack_depth
        self.kernel_size = int(config.get("kernel_size"))
        self.padding_size = int(config.get("padding_size"))
        self.rl_type = str(config.get("rl_type", "q"))
        assert self.rl_type == "v"

        self.use_batch_norm = int(config.get("use_batch_norm"))

        self.use_lstm = int(config.get("use_lstm", 0))
        self.use_rnn = int(config.get("use_rnn", 0))
        self.use_transformer = int(config.get("use_gtrxl", 0))
        self.use_transformer += int(config.get("use_transformer", 0))
        self.use_gru = int(config.get("use_gru", 0))
        self.use_gru += self.use_rnn
        self.use_all_rnn_layers = int(config.get("use_all_rnn_layers", 0))

        self.activation_function_string = config.get("activation_function", "relu")
        if self.activation_function_string == "relu":
            self.activation_fn = F.relu
        elif self.activation_function_string == "silu":
            self.activation_fn = F.silu

        if self.use_lstm or self.use_rnn:
            self.lstm_num_layers = int(config.get("lstm_num_layers"))
            self.lstm_num_directions = int(config.get("lstm_num_directions"))
            assert self.lstm_num_directions in (1, 2)
            self.lstm_is_bidirectional = bool(self.lstm_num_directions - 1)
            self.lstm_output_size = int(config.get("lstm_output_size"))

        input_channel_list: List = deepcopy(config.get("channel_list"))
        input_channel_list.insert(0, self.input_channels)
        layer_count = 0
        self.conv1 = nn.Conv2d(
            in_channels=input_channel_list[layer_count],
            out_channels=input_channel_list[layer_count + 1],
            kernel_size=self.kernel_size,
            padding=self.padding_size,
        )
        layer_count += 1

        if self.use_batch_norm:
            self.norm1 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.conv2 = nn.Conv2d(
            input_channel_list[layer_count],
            input_channel_list[layer_count + 1],
            kernel_size=self.kernel_size,
            padding=self.padding_size,
        )
        layer_count += 1

        if self.use_batch_norm:
            self.norm2 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.conv3 = nn.Conv2d(
            input_channel_list[layer_count],
            input_channel_list[layer_count + 1],
            kernel_size=self.kernel_size,
            padding=self.padding_size,
        )
        layer_count += 1

        if self.use_batch_norm:
            self.norm3 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.conv4 = nn.Conv2d(
            input_channel_list[layer_count],
            input_channel_list[layer_count + 1],
            kernel_size=self.kernel_size,
            padding=self.padding_size,
        )
        layer_count += 1
        if self.use_batch_norm:
            self.norm4 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.conv5 = nn.Conv2d(
            input_channel_list[layer_count],
            1,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

        if self.use_batch_norm:
            self.norm5 = nn.BatchNorm2d(1)

        # self.output_channels = input_channel_list[-1]
        self.output_channels = 1
        self.neurons_output = self.nr_actions_per_qubit * self.size * self.size + 1
        self.cnn_dimension = (self.size + 1) * (self.size + 1) * self.output_channels

        input_neuron_numbers = config["neuron_list"]

        print("Not using any recurrent module")
        lin_layer_count = 0
        self.lin0 = nn.Linear(self.cnn_dimension, int(input_neuron_numbers[0]))

        if self.use_batch_norm:
            self.norm5 = nn.BatchNorm1d(int(input_neuron_numbers[0]))

        self.lin1 = nn.Linear(
            input_neuron_numbers[lin_layer_count],
            input_neuron_numbers[lin_layer_count + 1],
        )
        lin_layer_count += 1

        if self.use_batch_norm:
            self.norm6 = nn.BatchNorm1d(input_neuron_numbers[lin_layer_count])

        self.output_layer = nn.Linear(int(input_neuron_numbers[-1]), 1)

    def forward(self, state: torch.Tensor):
        state = self._format(state)
        state = 2.0 * state - 1.0

        batch_size, _, _, _ = state.size()

        both = state.view(
            -1, self.input_channels, (self.size + 1), (self.size + 1)
        )  # convolve both

        both = self.conv1(both)
        if self.use_batch_norm:
            both = self.norm1(both)
        both = self.activation_fn(both)

        assert both.shape == (batch_size, 16, (self.size + 1), (self.size + 1))

        both = self.conv2(both)
        if self.use_batch_norm:
            both = self.norm2(both)
        both = self.activation_fn(both)

        both = self.conv3(both)
        if self.use_batch_norm:
            both = self.norm3(both)
        both = self.activation_fn(both)

        both = self.conv4(both)
        if self.use_batch_norm:
            both = self.norm4(both)
        both = self.activation_fn(both)

        both = self.conv5(both)
        if self.use_batch_norm:
            both = self.norm5(both)
        both = self.activation_fn(both)

        output = both.reshape(batch_size, -1)
        complete = self.lin0(output)

        if self.use_batch_norm:
            complete = self.activation_fn(self.norm5(complete))
        else:
            complete = self.activation_fn(complete)

        complete = self.lin1(complete)
        if self.use_batch_norm:
            complete = self.norm6(complete)
        complete = self.activation_fn(complete)

        final_output = self.output_layer(complete)
        return final_output
