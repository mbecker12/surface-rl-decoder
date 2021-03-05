import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from iniparser import Config
import random
import platform


class QuantumAgent1(nn.Module):

    def __init__(self):
        super(QuantumAgent1,self).__init__()
        c = Config()
        cwd = os.getcwd()
        _config = c.scan(cwd, True).read()
        config = c.config_rendered
        
        if "Windows" in platform.system():
            config_keyword = "src\\surface_rl_decoder\\config"
        else:
            config_keyword = "config"
        env_config = config.get(config_keyword)
        env_config = env_config.get("env")
        
        self.size = int(env_config.get("size"))
        syndrome_surface_size = (self.size+1)*(self.size+1)
        hidden_concat_size = int(env_config.get("hidden_concat_size"))
        self.nr_actions_per_qubit = int(env_config.get("nr_actions_per_qubit"))
        self.stack_depth = int(env_config.get("stack_depth"))

        self.epsilon_from = float(env_config.get("epsilon_from"))   
        self.epsilon_to = float(env_config.get("epsilon_to"))
        self.epsilon_decay = float(env_config.get("epsilon_decay"))
        self.gamma = float(env_config.get("gamma"))
        self.lstm_layers = int(env_config.get("lstm_layers"))
        hiddenX = int(env_config.get("hidden_x"))
        hiddenZ = int(env_config.get("hidden_z"))
        hiddenBoth = int(env_config.get("hidden_both"))
        #self.step = 0

        self.inputLayerX = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenX)
        self.inputLayerBoth = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenBoth)
        self.inputLayerZ = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenZ)
        self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, hidden_concat_size)
        self.concatenatedComplete = nn.Linear(hidden_concat_size+hiddenBoth, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        self.lstmLayer = nn.LSTM(self.nr_actions_per_qubit*(self.size)*(self.size)+1, self.nr_actions_per_qubit*(self.size)*(self.size)+1, num_layers=3, bidirectional = True)
        self.final_layer = nn.Linear((self.nr_actions_per_qubit*(self.size)*(self.size)+1)*2, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        



    def forward(self, x, z, both, batch_size = 1): #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = x.view(self.stack_depth, batch_size, (self.size+1)*(self.size+1), 1)
        x = torch.squeeze(x)
        x = F.relu(self.inputLayerX(x))
        z = z.view(self.stack_depth, batch_size, (self.size+1)*(self.size+1), 1)
        z = torch.squeeze(z)
        z = F.relu(self.inputLayerZ(z))
        both = both.view(self.stack_depth, batch_size, (self.size+1)*(self.size+1), 1)
        both = torch.squeeze(both)
        both = F.relu(self.inputLayerBoth(both))
        xz = torch.cat((x,z), 2)
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat((xz,both), 2)
        complete = F.sigmoid(self.concatenatedComplete(complete))
        h = torch.zeros(self.lstm_layers*2, batch_size, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        c = torch.zeros(self.lstm_layers*2, batch_size, self.nr_actions_per_qubit*(self.size)*(self.size)+1)
        for i in range(self.stack_depth):
            output, (h, c) = lstmLayer(complete[i,:].view(self.stack_depth, batch_size, self.nr_actions_per_qubit*(self.size)*(self.size)+1), h, c)

        output = self.final_layer(output)
        #agent.step += 1
        return F.softmax(output)

