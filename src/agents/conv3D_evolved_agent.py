"""
Implementation of an agent containing 3D convolutional layers
followed by linear layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.interface import interface
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class Conv3DEvolvedAgent(nn.Module):

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
        channel_list: list of integers which hold the number of input channels for each layer
            and the final number being the number of output channels from the final conv3d layer
        second_channel_list: similar to above but for the combination of the different output from x, y, both syndrome sections,
            also, note that in this case, the numbers are the output from the previous layers rather thn the input as
            the last element of channel_list decides the number of input channels into these layers.
        neurons_lin_layer: number of neurons for the second-to-last linear layer

        kernel_size: the size of the kernel
        padding_size: the amount of padding,
            should be a number such that the shortening in dimension length
            due to the kernel is negated
    """

    def __init__(self, config):
        super(Conv3DEvolvedAgent, self).__init__()

        self.device = config.get("device")
        # pylint: disable=not-callable
        self.plaquette_mask = torch.tensor(plaquette_mask, device=self.device)
        self.vertex_mask = torch.tensor(vertex_mask, device=self.device)
        self.split_input_toggle = int(config.get("split_input_toggle", 1))

        self.size = int(config.get("code_size"))
        self.nr_actions_per_qubit = int(config.get("num_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))
        
        self.padding_size = int(config.get("padding_size"))
        self.kernel_size = int(config.get("kernel_size"))
        self.kernel_depth = int(config.get("kernel_depth", self.kernel_size))
        self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_depth)
        
        self.neurons_lin_layer = int(config.get("neurons_lin_layer"))

        module_list = config.get("channel_list")
        modules = []
        for i in range(len(module_list)-1):
            modules.append(nn.Conv3d(i, i+1, self.kernel_size, padding = self.padding_size))
        self.sequential_x = nn.Sequential(*modules)
        self.sequential_z = nn.Sequential(*modules)
        self.sequential_both = nn.Sequential(*modules)
        self.output_channels = module_list[-1] #for the view in the run to ascertain the tensor shape

        second_module_list = config.get("second_channel_list", [])
        
        if  len(second_module_list) > 2:
            second_modules = []
            self.second_module_flag = True
            second_modules.append(nn.Conv3d(module_list[-1], second_module_list[0], self.kernel_size, padding = self.padding_size))
            self.last_output_channels = second_module_list[-1]
            for i in range(1, len(second_module_list)):
                second_modules.append(nn.Conv3d(i-1, i,self.kernel_size, padding = self.padding_size))
            self.sequential_complete = nn.Sequential(*modules)
            
        else:
            self.second_module_flag = False

        if not self.second_module_flag:
            self.almost_final_layer = nn.Linear(
                (self.size + 1) * (self.size + 1) * self.module_list[-1],
                self.neurons_lin_layer,
            )
        else:
            self.almost_final_layer = nn.Linear(
                (self.size + 1) * (self.size + 1) * self.second_module_list[-1],
                self.neurons_lin_layer
            )

        self.final_layer = nn.Linear(
            self.stack_depth * self.neurons_lin_layer,
            self.nr_actions_per_qubit * (self.size) * (self.size) + 1,
        )

        self.input_channels = module_list[0]

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
            x = self.sequential_x(x)

            z = z.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve z
            z = self.sequential_z(z)


            both = both.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve both
            both = self.sequential_both(both)

            complete = (x + z + both) / 3  # add them together
            
            
            
        else:
            both = state.view(
                -1,
                self.input_channels,
                self.stack_depth,
                (self.size + 1),
                (self.size + 1),
            )  # convolve both
            complete = self.sequential_both(both)


        #assert complete.shape == [-1, self.stack_depth, (self.size + 1)*(self.size + 1)*self.output_channels] #dunno if this works
            
        if self.second_module_flag:
            complete = self.sequential_complete(complete)
            self.output_channels = self.last_output_channels 

        complete = complete.view(-1, self.stack_depth*(self.size + 1)*(self.size + 1)*self.output_channels)

        complete = F.relu(self.almost_final_layer(complete))
        
        # shift the dimension so that
        # dimension -1 gives us a matrix/vector
        # with regards to batch and actions for each of those batches
        #complete = complete.view(
        #    -1,
        #    self.stack_depth * self.neurons_lin_layer
        #)
        final_output = self.final_layer(complete)

        return final_output
