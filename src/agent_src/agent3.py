import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class QuantumAgent3(nn.Module):

    def __init__(self, config):
        super(QuantumAgent3,self).__init__()
        
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
        self.lstm_layers = int(config.get("lstm_layers"))



        self.input_conv_layerX = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, padding = self.padding_size)
        self.input_conv_layerBoth = nn.Conv2d(self.input_channels,self.output_channels, self.kernel_size, padding = self.padding_size)
        self.input_conv_layerZ = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, padding = self.padding_size)

        self.nd_conv_layerX = nn.Conv2d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)
        self.nd_conv_layerZ = nn.Conv2d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)
        self.nd_conv_layerBoth = nn.Conv2d(self.output_channels, self.output_channels2, self.kernel_size, padding = self.padding_size)

        self.rd_conv_layerX = nn.Conv2d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)
        self.rd_conv_layerZ = nn.Conv2d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)
        self.rd_conv_layerBoth = nn.Conv2d(self.output_channels2, self.output_channels3, self.kernel_size, padding = self.padding_size)

        self.comp_conv_layerX = nn.Conv2d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)
        self.comp_conv_layerZ = nn.Conv2d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)
        self.comp_conv_layerBoth = nn.Conv2d(self.output_channels3, 1, self.kernel_size, padding = self.padding_size)

        self.lstm_layer = nn.LSTM((self.size+1)*(self.size+1), self.nr_actions_per_qubit*(self.size)*(self.size)+1, num_layers=self.lstm_layers, bidirectional = True)        
        
        self.almost_final_layer = nn.Linear((self.nr_actions_per_qubit*(self.size)*(self.size)+1)*2, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        self.final_layer = nn.Linear(self.nr_actions_per_qubit*(self.size)*(self.size)+1, self.nr_actions_per_qubit*(self.size)*(self.size)+1)



    def forward(self, state):
        x, z, both = self.interface(state) #multiple input channels for different procedures, they are then concatenated as the data is processed

        x = x.view(-1, self.input_channels, (self.size+1), (self.size+1))
        x = F.relu(self.input_conv_layerX(x))
        x = F.relu(self.nd_conv_layerX(x))
        x = self.rd_conv_layerX(x)
        x = self.comp_conv_layerX(x)

        z = z.view(-1, self.input_channels, (self.size+1), (self.size+1))
        z = F.relu(self.input_conv_layerZ(z))
        z = F.relu(self.nd_conv_layerZ(z))
        z = self.rd_conv_layerZ(z)
        z = self.comp_conv_layerZ(z)

        both = both.view(-1, self.input_channels, (self.size+1), (self.size+1))
        both = F.relu(self.input_conv_layerBoth(both))
        both = F.relu(self.nd_conv_layerBoth(both))
        both = self.rd_conv_layerBoth(both)
        both = self.comp_conv_layerBoth(both)

        complete = (x+z+both)/3
        complete = complete.view(self.stack_depth, -1,  (self.size+1)*(self.size+1))
        output, (_h,_c) = self.lstm_layer(complete)

        output = self.almost_final_layer(output[-1])
        final_output = self.final_layer(output)
        
        
        return final_output

    def interface(self, state):
        x = state*plaquette_mask
        z = state*vertex_mask
        return x, z, state