
import torch
import numpy as np
from src.distributed.model_util import extend_model_config, choose_model
from src.agents.conv3D_evolved_agent import Conv3DEvolvedAgent


def test_conv3d_evolved_agent:
    