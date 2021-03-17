import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class QuantumAgent2(nn.Module):

    def __init__(self, config):
        super(QuantumAgent1,self).__init__()
        
        self.size = int(config.get("size"))
        syndrome_surface_size = (self.size+1)*(self.size+1)
        hidden_concat_size = int(config.get("hidden_concat_size"))
        self.nr_actions_per_qubit = int(config.get("nr_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))

        self.epsilon_from = float(config.get("epsilon_from"))   
        self.epsilon_to = float(config.get("epsilon_to"))
        self.epsilon_decay = float(config.get("epsilon_decay"))

        input_channels = int(config.get("input_channels"))
        self.kernel_size = int(config.get("kernel_size"))
        self.output_channels = int(config.get("output_channels"))
        self.output_channels2 = int(config.get("output_channels2"))
        self.output_channels3 = int(config.get("output_channels3"))
        self.padding_size = int(config.get("padding_size"))



        self.input_conv_layerX = nn.Conv3D(1, self.output_channels, kernel_size, padding = self.padding_size)
        self.input_conv_layerBoth = nn.Conv3d(1,self.out_channels, kernel_size, padding = self.padding_size)
        self.input_conv_layerZ = nn.Conv3d(1, self.out_channels, kernel_size, padding = self.padding_size)

        self.2nd_conv_layerX = nn.Conv3D(self.out_channels, self.out_channels2, kernel_size, padding = self.padding_size)
        self.2nd_conv_layerZ = nn.Conv3D(self.output_channels, self.output_channels2, kernel_size, padding = self.padding_size)
        self.2nd_conv_layerBoth = nn.Conv3D(self.output_channels, self.output_channels2, kernel_size, padding = self.padding_size)

        self.3rd_conv_layerX = nn.Conv3D(self.output_channels2, self.output_channels3, kernel_size, padding = self.padding_size)
        self.3rd_conv_layerZ = nn.Conv3D(self.output_channels2, self.output_channels3, kernel_size, padding = self.padding_size)
        self.3rd_conv_layerBoth = nn.Conv3D(self.output_channels2, self.output_channels3, kernel_size, padding = self.padding_size)

        self.comp_conv_layerX = nn.Conv3D(self.output_channels3, 1, kernel_size, padding = self.padding_size)
        self.comp_conv_layerZ = nn.Conv3D(self.output_channels3, 1, kernel_size, padding = self.padding_size)
        self.comp_conv_layerBoth = nn.Conv3D(self.output_channels3, 1, kernel_size, padding = self.padding_size)

        self.final_layer = nn.Linear(( (self.size+1)*(self.size+1), self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        



    def forward(self, state): 
        x, z, both = self.interface(state) #multiple input channels for different procedures, they are then concatenated as the data is processed

        x = x.view(-1, 1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        x = F.relu(self.input_conv_layerX(x))
        x = F.relu(self.2nd_conv_layerX(x))
        x = self.3rd_conv_layerX(x)
        x = self.comp_conv_layerX(x)

        z = z.view(-1, 1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        z = F.relu(self.input_conv_layerZ(z))
        z = F.relu(self.2nd_conv_layerZ(z))
        z = self.3rd_conv_layerZ(z)
        z = self.comp_conv_layerZ(z)

        both = both.view(-1, 1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        both = F.relu(self.input_conv_layerBoth(z))
        both = F.relu(self.2nd_conv_layerBoth(z))
        both = self.3rd_conv_layerBoth(z)
        both = self.comp_conv_layerBoth(z)

        complete = (x+z+both)/3
        complete = complete.view(self.stack_depth, -1,  (self.size+1)*(self.size+1))
        final_output = self.final_layer(complete)
        
        
        return final_output

    def interface(self, state):
        x = state*plaquette_mask
        z = state*vertex_mask
        return x, z, state