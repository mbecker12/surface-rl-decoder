from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from surface_rl_decoder.syndrome_masks import get_vertex_mask, get_plaquette_mask

import logging

logger = logging.getLogger(name="MODEL")


class Conv2dAgentFullyConv(BaseAgent):
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
        self.rl_type = str(config.get("rl_type", "v"))
        assert self.rl_type == "q"

        self.use_batch_norm = int(config.get("use_batch_norm"))

        self.output_neurons = 3 * self.size * self.size + 1

        self.activation_function_string = config.get("activation_function", "relu")
        if self.activation_function_string == "relu":
            self.activation_fn = F.relu
        elif self.activation_function_string == "silu":
            self.activation_fn = F.silu

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
            input_channel_list[layer_count + 1],
            kernel_size=(1, 1),
            padding=(0, 0),
        )

        layer_count += 1
        if self.use_batch_norm:
            self.norm5 = nn.BatchNorm2d(input_channel_list[layer_count])

        self.conv6 = nn.Conv2d(
            input_channel_list[layer_count],
            input_channel_list[layer_count + 1],
            kernel_size=(1, 1),
            padding=(0, 0),
        )

        layer_count += 1

        # TODO: could also try this here with (3, 3) kernel w/o padding
        # instead of pooling [for operator channels]
        self.action_conv = nn.Conv2d(
            input_channel_list[layer_count], 4, kernel_size=(3, 3), padding=(1, 1)
        )

        self.operator_pool = nn.AdaptiveAvgPool2d(output_size=(self.size, self.size))

        self.terminal_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.output_channels = input_channel_list[-1]
        self.output_channels = 4
        self.neurons_output = self.nr_actions_per_qubit * self.size * self.size + 1
        self.cnn_dimension = (self.size + 1) * (self.size + 1) * self.output_channels

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

        both = self.conv6(both)
        if self.use_batch_norm:
            both = self.norm6(both)
        both = self.activation_fn(both)

        both = self.action_conv(both)

        pooled_operators = self.operator_pool(both[:, :3, :, :])
        pooled_operators_permuted = pooled_operators.permute(0, 3, 2, 1).contiguous()
        final_operators = pooled_operators_permuted.view(batch_size, -1)

        tmp_terminal = both[:, -1, ::].unsqueeze(1)
        pooled_terminal = self.terminal_pool(tmp_terminal)
        final_terminal = pooled_terminal.reshape(batch_size, 1)

        final_tensor = torch.cat((final_operators, final_terminal), dim=1)

        # for channel in range(batch_size):
        #     for x in range(5):
        #         for y in range(5):
        #             for ac in range(3):
        #                 index = x * 3 + y * self.size * 3 + ac
        #                 assert (
        #                     pooled_operators[channel, ac, x, y]
        #                     == final_tensor[channel, index]
        #                 ), f"\n{index=}, {pooled_operators[channel, x, y, ac]=}, {final_tensor[channel, index]=}\n\n{final_tensor[channel]=}\n\n{pooled_operators[channel]=}\n\n{pooled_operators_permuted[channel]=}"

        return final_tensor
