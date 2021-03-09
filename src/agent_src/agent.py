import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_rl_decoder.syndrome_masks import plaquette_mask, vertex_mask


class QuantumAgent1(nn.Module):

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
        
        self.lstm_layers = int(config.get("lstm_layers"))
        hiddenX = int(config.get("hidden_x"))
        hiddenZ = int(config.get("hidden_z"))
        hiddenBoth = int(config.get("hidden_both"))

        self.inputLayerX = nn.Linear(syndrome_surface_size,hiddenX)
        self.inputLayerBoth = nn.Linear(syndrome_surface_size,hiddenBoth)
        self.inputLayerZ = nn.Linear(syndrome_surface_size,hiddenZ)
        self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, hidden_concat_size)
        self.concatenatedComplete = nn.Linear(hidden_concat_size+hiddenBoth, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        self.lstmLayer = nn.LSTM(self.nr_actions_per_qubit*(self.size)*(self.size)+1, self.nr_actions_per_qubit*(self.size)*(self.size)+1, num_layers=self.lstm_layers, bidirectional = True)
        self.final_layer = nn.Linear((self.nr_actions_per_qubit*(self.size)*(self.size)+1)*2, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        



    def forward(self, state): 
        x, z, both = self.interface(state) #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = x.view(self.stack_depth, -1, (self.size+1)*(self.size+1), 1)
        x = torch.squeeze(x)
        x = F.relu(self.inputLayerX(x))
        z = z.view(self.stack_depth, -1, (self.size+1)*(self.size+1), 1)
        z = torch.squeeze(z)
        z = F.relu(self.inputLayerZ(z))
        both = both.view(self.stack_depth, -1, (self.size+1)*(self.size+1), 1)
        both = torch.squeeze(both)
        both = F.relu(self.inputLayerBoth(both))
        xz = torch.cat((x,z), -1)
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat((xz,both), -1)
        complete = F.sigmoid(self.concatenatedComplete(complete))
        complete = complete.view(self.stack_depth, -1, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        
        output, (_h, _c) = self.lstmLayer(complete)
        final_output = self.final_layer(output[-1])

        return final_output

    def interface(self, state):
        x = state*plaquette_mask
        z = state*vertex_mask
        return x, z, state