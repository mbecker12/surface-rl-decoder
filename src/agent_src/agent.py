import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class QuantumAgent1(nn.Module):

    """
    Description:
        First iteration of an agent. Consists of multiple linear layers that takes the input. 
        Splits the input into x,z and both errors which it then proceeds to feed each part into a neural network
        before concatenating them and feeding them together into more linear layers and an LSTM layer from which only the final output is used.

        For the instantiation it requires a dictionary containing the several parameters that the network will need to build itself.

    Parameters
    ==========
        size: the length of the surface code
        nr_actions_per_cubit: in most cases should be 3 but can be adjusted, it is the number of types of correction actions that can be applied on the qubit
        stack_depth: the length of the dimension with aspect to time

        hiddenX: the size of the hidden layer for which the X syndrome propagates through
        hiddenZ: the size of the hidden layer for which the Z syndrome propagates through
        hiddenBoth: the size of the hidden layer for which both syndromes propagates through
        lstm_layers: the lstm network has a parameter which queries for how many lstm layers should be stacked on top of each other
        


    """

    def __init__(self, config):
        super(QuantumAgent1,self).__init__()
        
        self.size = int(config.get("size"))
        syndrome_surface_size = (self.size+1)*(self.size+1)
        hidden_concat_size = int(config.get("hidden_concat_size"))
        self.nr_actions_per_qubit = int(config.get("nr_actions_per_qubit"))
        self.stack_depth = int(config.get("stack_depth"))
        
        self.lstm_layers = int(config.get("lstm_layers"))
        hiddenX = int(config.get("hidden_x"))
        hiddenZ = int(config.get("hidden_z"))
        hiddenBoth = int(config.get("hidden_both"))

        self.input_layerX = nn.Linear(syndrome_surface_size,hiddenX)
        self.input_layerBoth = nn.Linear(syndrome_surface_size,hiddenBoth)
        self.input_layerZ = nn.Linear(syndrome_surface_size,hiddenZ)
        self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, hidden_concat_size)
        self.concatenatedComplete = nn.Linear(hidden_concat_size+hiddenBoth, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        self.lstm_layer = nn.LSTM(self.nr_actions_per_qubit*(self.size)*(self.size)+1, self.nr_actions_per_qubit*(self.size)*(self.size)+1, num_layers=self.lstm_layers, bidirectional = True)
        self.final_layer = nn.Linear((self.nr_actions_per_qubit*(self.size)*(self.size)+1)*2, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        



    def forward(self, state): 
        x, z, both = self.interface(state)          #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = x.view(-1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        x = torch.squeeze(x)
        x = F.relu(self.input_layerX(x))
        z = z.view(-1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        z = torch.squeeze(z)
        z = F.relu(self.input_layerZ(z))
        both = both.view(-1, self.stack_depth, (self.size+1)*(self.size+1), 1)
        both = torch.squeeze(both)
        both = F.relu(self.input_layerBoth(both))
        xz = torch.cat((x,z), -1)                    #concatenate x and z           
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat((xz,both), -1)         #concatenate all the data
        complete = F.sigmoid(self.concatenatedComplete(complete))
        complete = complete.view(self.stack_depth, -1, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        
        output, (_h, _c) = self.lstm_layer(complete)
        final_output = self.final_layer(output[-1]) #only use the last output from the lstm to pick an action (we want to correct errors on the last time layer)

        return final_output

    def interface(self, state): #help function to split up the input into the three channels
        x = state*plaquette_mask
        z = state*vertex_mask
        return x, z, state