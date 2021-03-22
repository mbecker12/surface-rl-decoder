import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask

#QuantumAgent2
class Conv3dAgent(nn.Module):

    """
    Description:
        Second iteration of an agent. Consists of multiple 3D convolutional layers. Splits the input into x,z and both errors which it then proceeds to feed each part into a neural network
        before convolving them to a single feature map and adding them together and feeding them together into 2 linear layers from which only the final (with regards to time) output is used.

        For the instantiation it requires a dictionary containing the several parameters that the network will need to build itself.

    Parameters
    ==========
        size: the length of the surface code
        nr_actions_per_cubit: in most cases should be 3 but can be adjusted, it is the number of types of correction actions that can be applied on the qubit
        stack_depth: the length of the dimension with aspect to time

        input_channels: the size of the number of channels at the input. Should be 1 in most cases but can be adjusted if one wishes
        output_channels: the number of output channels from the first 3d convolution
        output_channels2: the number of output channels from the second 3d convolution
        output_channels3: the number of output channels from the third 3d convolution
        
        kernel_size: the size of the kernel
        padding_size: the amount of padding, should be a number such that the shortening in dimension length due to the kernel is negated

    """

    def __init__(self, config):
        super(QuantumAgent2,self).__init__()
        
        self.size = int(config.get("size"))
        syndrome_surface_size = (self.size+1)*(self.size+1)
        self.nr_actions_per_qubit = int(config.get("nr_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))

        self.input_channels = int(config.get("input_channels"))
        self.kernel_size = int(config.get("kernel_size"))
        self.output_channels = int(config.get("output_channels"))
        self.output_channels2 = int(config.get("output_channels2"))
        self.output_channels3 = int(config.get("output_channels3"))
        self.padding_size = int(config.get("padding_size"))



        self.input_conv_layerX = nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size, padding = self.padding_size)
        self.input_conv_layerBoth = nn.Conv3d(self.input_channels,self.output_channels, self.kernel_size, padding = self.padding_size)
        self.input_conv_layerZ = nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size, padding = self.padding_size)

        self.nd_conv_layerX = nn.Conv3d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)
        self.nd_conv_layerZ = nn.Conv3d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)
        self.nd_conv_layerBoth = nn.Conv3d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)

        self.rd_conv_layerX = nn.Conv3d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)
        self.rd_conv_layerZ = nn.Conv3d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)
        self.rd_conv_layerBoth = nn.Conv3d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)

        self.comp_conv_layerX = nn.Conv3d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)
        self.comp_conv_layerZ = nn.Conv3d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)
        self.comp_conv_layerBoth = nn.Conv3d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)


        
        self.almost_final_layer = nn.Linear((self.size+1)*(self.size+1) , self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        self.final_layer = nn.Linear(self.nr_actions_per_qubit*(self.size)*(self.size)+1,self.nr_actions_per_qubit*(self.size)*(self.size)+1)



    def forward(self, state):
        x, z, both = self.interface(state) #multiple input channels for different procedures, they are then concatenated as the data is processed

        x = x.view(-1, self.input_channels, self.stack_depth, (self.size+1), (self.size+1)) #convolve x
        x = F.relu(self.input_conv_layerX(x))
        x = F.relu(self.nd_conv_layerX(x))
        x = self.rd_conv_layerX(x)
        x = self.comp_conv_layerX(x)

        z = z.view(-1, self.input_channels, self.stack_depth, (self.size+1), (self.size+1)) #convolve z
        z = F.relu(self.input_conv_layerZ(z))
        z = F.relu(self.nd_conv_layerZ(z))
        z = self.rd_conv_layerZ(z)
        z = self.comp_conv_layerZ(z)

        both = both.view(-1, self.input_channels, self.stack_depth, (self.size+1), (self.size+1))   #convolve the combination
        both = F.relu(self.input_conv_layerBoth(both))
        both = F.relu(self.nd_conv_layerBoth(both))
        both = self.rd_conv_layerBoth(both)
        both = self.comp_conv_layerBoth(both)

        complete = (x+z+both)/3     #add them together (and average them)
        complete = complete.view(-1, self.stack_depth,  (self.size+1)*(self.size+1)) # make sure the dimensions are in order
        complete = self.almost_final_layer(complete)
        complete = complete.view(self.stack_depth, -1, self.nr_actions_per_qubit*(self.size)*(self.size)+1) #shift the dimension so that
        final_output = self.final_layer(complete[-1])   #dimension -1 gives us a matrix/vector with regards to batch and actions for each of those batches
        
        
        return final_output

    def interface(self, state): #help function to split up the input into the three channels
        x = state*plaquette_mask
        z = state*vertex_mask
        return x, z, state