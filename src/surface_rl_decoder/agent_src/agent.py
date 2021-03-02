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
    size = int(env_config.get("size"))
    surface_size = size*size
    syndrome_surface_size = (size+1)*(size+1)
    hidden_concat_size = int(env_config.get("hidden_concat_size"))
    nr_actions = int(env_config.get("nr_actions_per_qubit"))

    self.epsilon_from = double(env_config.get("epsilon_from"))   
    self.epsilon_to = double(env_config.get("epsilon_to"))
    self.epsilon_decay = double(env_config.get("epsilon_decay"))
    self.gamma = double(env_config.get("gamma"))
    self.step = 0


    super(QuantumAgent1,self).__init__()
    self.inputLayerX = nn.Linear(syndrome_surface_size,hiddenX)
    self.inputLayerBoth = nn.Linear(syndrome_surface_size,hiddenBoth)
    self.inputLayerZ = nn.Linear(syndrome_surface_size,hiddenZ)
    self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, hidden_concat_size)
    self.concatenatedComplete = nn.Linear(hidden_concat_size+hiddenBoth, nr_actions_per_qubit*(size)*(size)+1)
    



    def forward(self,x z, both): #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = F.relu(self.inputLayerX(x))
        z = F.relu(self.inputLayerZ(z))
        both = F.relu(self.inputLayerBoth(both))
        xz = torch.cat(x,z)
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat(xz,both)
        complete = F.sigmoid(self.concatenatedComplete(complete))
        
        agent.step += 1
        return complete

