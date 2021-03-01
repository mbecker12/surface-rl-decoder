import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from iniparser import Config


def configuration()
    cfg = Config()
    cwd = os.getcwd()
    _config = cfg.scan(cwd, True).read()
    config = cfg.config_rendered.get("config")

    env_config = config.get("env")




class QuantumAgent1(nn.module)

    def __init__(self, hiddenX,hiddenZ,hiddenBoth):

        Config()
        super(QuantumAgent1,self).__init__()
        self.inputLayerX = nn.Linear(SYNDROME_SURFACE_SIZE,hiddenX)
        self.inputLayerBoth = nn.Linear(SYNDROME_SURFACE_SIZE,hiddenBoth)
        self.inputLayerZ = nn.Linear(SYNDROME_SURFACE_SIZE,hiddenZ)
        self.concatenatedXZ = nn.Linear(hiddenZ+hiddenX, HIDDEN_CONCAT_SIZE)
        self.concatenatedComplete = nn.Linear(HIDDEN_CONCAT_SIZE+hiddenBoth, 3*(size))
        self.eps = EPSILON_FROM



    def forward(self,x z, both)  #multiple input channels for different procedures, they are then concatenated as the data is processed
        x = F.relu(self.inputLayerX(x))
        z = F.relu(self.inputLayerZ(z))
        both = F.relu(self.inputLayerBoth(both))
        xz = torch.cat(x,z)
        xz = F.relu(self.concatenatedXZ(xz))
        complete = torch.cat(xz,both)
        complete = F.relu(self.concatenatedComplete(complete))

        return complete
