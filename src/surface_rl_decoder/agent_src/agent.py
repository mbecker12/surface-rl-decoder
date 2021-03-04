import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from iniparser import Config
import random


class QuantumAgent1(nn.module)

    def __init__(self, hiddenX,hiddenZ,hiddenBoth):

    cfg = Config()
    _config = cfg.scan(cwd, True).read()
    self.config = cfg.config_rendered.get("config")

    env_config = config.get("env")
    self.size = int(env_config.get("size"))
    surface_size = self.size*self.size
    syndrome_surface_size = (self.size+1)*(self.size+1)
    hidden_concat_size = int(env_config.get("hidden_concat_size"))
    nr_actions = int(env_config.get("nr_actions_per_qubit"))
    self.stack_depth = int(env_config.get("stack_depth"))

    self.epsilon_from = double(env_config.get("epsilon_from"))   
    self.epsilon_to = double(env_config.get("epsilon_to"))
    self.epsilon_decay = double(env_config.get("epsilon_decay"))
    self.gamma = double(env_config.get("gamma"))
    self.lstm_layers = int(env_config.get("lstm_layers"))
    self.step = 0


    super(QuantumAgent1,self).__init__()
    self.inputLayerX = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenX)
    self.inputLayerBoth = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenBoth)
    self.inputLayerZ = nn.Linear(syndrome_surface_size*syndrome_surface_size,hiddenZ)
    self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, hidden_concat_size)
    self.concatenatedComplete = nn.Linear(hidden_concat_size+hiddenBoth, nr_actions_per_qubit*(self.size)*(self.size)+1)
    self.lstmLayer = nn.LSTM(nr_actions_per_qubit*(self.size)*(self.size)+1,nr_actions_per_qubit*(self.size)*(self.size)+1, num_layers=3, bidirectional = True)
    self.final_layer = nn.Linear((nr_actions_per_qubit*(self.size)*(self.size)+1)*2, nr_actions_per_qubit*(self.size)*(self.size)+1)
    



    def forward(self, x, z, both): #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = x.view(self.stack_depth, (self.size+1)*(self.size+1), 1)
        x = torch.squeeze(x)
        x = F.relu(self.inputLayerX(x))
        z = z.view(self.stack_depth, (self.size+1)*(self.size+1), 1)
        z = torch.squeeze(z)
        z = F.relu(self.inputLayerZ(z))
        both = both.view(self.stack_depth, (self.size+1)*(self.size+1), 1)
        both = torch.squeeze(both)
        both = F.relu(self.inputLayerBoth(both))
        xz = torch.cat((x,z),1)
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat((xz,both),1)
        complete = F.sigmoid(self.concatenatedComplete(complete))
        h = torch.zeros(self.lstm_layers*2, nr_actions_per_qubit*(self.size)*(self.size)+1)
        c = torch.zeros(self.lstm_layers*2, nr_actions_per_qubit*(self.size)*(self.size)+1)

        for i in range(self.stack_depth):
            output, (h, c) = lstmLayer(complete[i,:], h, c)

        output = self.final_layer(output)
        agent.step += 1
        return F.softmax(output)

