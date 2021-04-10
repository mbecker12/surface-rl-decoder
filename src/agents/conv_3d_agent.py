"""
Implementation of an agent containing 3D convolutional layers
followed by linear layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.interface import interface
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class Conv3dGeneralAgent(nn.Module):

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
        super(Conv3dGeneralAgent, self).__init__()

        self.device = config.get("device")
        # pylint: disable=not-callable
        self.plaquette_mask = torch.tensor(plaquette_mask, device=self.device)
        self.vertex_mask = torch.tensor(vertex_mask, device=self.device)
        self.split_input_toggle = int(config.get("split_input_toggle", 1))

        self.size = int(config.get("code_size"))
        self.nr_actions_per_qubit = int(config.get("num_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))

        self.input_channels = int(config.get("input_channels"))
        self.output_channels = int(config.get("output_channels"))
        self.output_channels2 = int(config.get("output_channels2"))
        self.output_channels3 = int(config.get("output_channels3"))
        self.output_channels4 = int(config.get("output_channels4"))

        self.neurons_lin_layer = int(config.get("neurons_lin_layer"))

        self.padding_size = int(config.get("padding_size"))
        self.kernel_size = int(config.get("kernel_size"))
        self.kernel_depth = int(config.get("kernel_depth", self.kernel_size))
        self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_depth)

        self.input_conv_layer_both = nn.Conv3d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.nd_conv_layer_both = nn.Conv3d(
            self.output_channels,
            self.output_channels2,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.rd_conv_layer_both = nn.Conv3d(
            self.output_channels2,
            self.output_channels3,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.comp_conv_layer_both = nn.Conv3d(
            self.output_channels3,
            self.output_channels4,
            self.kernel_size,
            padding=self.padding_size,
        )

        self.input_conv_layer_x = nn.Conv3d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.nd_conv_layer_x = nn.Conv3d(
            self.output_channels,
            self.output_channels2,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.rd_conv_layer_x = nn.Conv3d(
            self.output_channels2,
            self.output_channels3,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.comp_conv_layer_x = nn.Conv3d(
            self.output_channels3,
            self.output_channels4,
            self.kernel_size,
            padding=self.padding_size,
        )

        self.input_conv_layer_z = nn.Conv3d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.nd_conv_layer_z = nn.Conv3d(
            self.output_channels,
            self.output_channels2,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.rd_conv_layer_z = nn.Conv3d(
            self.output_channels2,
            self.output_channels3,
            self.kernel_size,
            padding=self.padding_size,
        )
        self.comp_conv_layer_z = nn.Conv3d(
            self.output_channels3,
            self.output_channels4,
            self.kernel_size,
            padding=self.padding_size,
        )

        self.almost_final_layer = nn.Linear(
            (self.size + 1) * (self.size + 1) * self.output_channels4,
            self.neurons_lin_layer,
        )
        self.final_layer = nn.Linear(
            self.stack_depth * self.neurons_lin_layer,
            self.nr_actions_per_qubit * (self.size) * (self.size) + 1,
        )

    def forward(self, state: torch.Tensor):
        """
        forward pass
        """

        if self.split_input_toggle:
            x, z, both = interface(state, self.plaquette_mask, self.vertex_mask)
            # multiple input channels for different procedures,
            # they are then concatenated as the data is processed

            x = x.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve x
            x = F.relu(self.input_conv_layer_x(x))
            x = F.relu(self.nd_conv_layer_x(x))
            x = F.relu(self.rd_conv_layer_x(x))
            x = F.relu(self.comp_conv_layer_x(x))

            z = z.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve z
            z = F.relu(self.input_conv_layer_z(z))
            z = F.relu(self.nd_conv_layer_z(z))
            z = F.relu(self.rd_conv_layer_z(z))
            z = F.relu(self.comp_conv_layer_z(z))

            both = both.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve both
            both = F.relu(self.input_conv_layer_both(both))
            both = F.relu(self.nd_conv_layer_both(both))
            both = F.relu(self.rd_conv_layer_both(both))
            both = F.relu(self.comp_conv_layer_both(both))

            complete = (x + z + both) / 3  # add them together
            complete = complete.view(
                -1,
                self.stack_depth,
                (self.size + 1) * (self.size + 1) * self.output_channels4,
            )  # make sure the dimensions are in order
        else:
            both = state.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve both
            both = F.relu(self.input_conv_layer_both(both))
            both = F.relu(self.nd_conv_layer_both(both))
            both = F.relu(self.rd_conv_layer_both(both))
            both = F.relu(self.comp_conv_layer_both(both))

            complete = both.view(
                -1,
                self.stack_depth,
                (self.size + 1) * (self.size + 1) * self.output_channels4,
            )  # make sure the dimensions are in order

        complete = F.relu(self.almost_final_layer(complete))
        # shift the dimension so that
        # dimension -1 gives us a matrix/vector
        # with regards to batch and actions for each of those batches
        complete = complete.view(
            -1,
            self.stack_depth * self.neurons_lin_layer,
        )
        final_output = self.final_layer(complete)

        return final_output
