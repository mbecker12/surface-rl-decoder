import os
import yaml
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.analysis_util import load_analysis_model

from analysis.training_run_class import TrainingRun

base_model_config_path = "src/config/model_spec/old_conv_agents.json"
base_model_path = "remote_networks/5/65280/simple_conv_5_65280.pt"


training_runs = [
    TrainingRun(69037, 5, 5, 0.0108, 0.0, "q", "3D Conv", model_name="conv3d"),
    TrainingRun(
        71852,
        5,
        5,
        0.008,
        0.008,
        "q",
        "2D Conv + GRU",
        model_name="conv2d",
        model_config_file="conv_agents_slim_gru.json",
        transfer_learning=1,
    ),
    TrainingRun(69312, 5, 5, 0.01, 0.01, "q", "3D Conv", model_name="conv3d"),
    TrainingRun(71873, 5, 5, 0.01, 0.01, "ppo", "3D Conv", model_name="conv3d"),
    TrainingRun(72409, 5, 5, 0.01, 0.01, "q", "2D Conv", model_name="conv2d"),
]

plt.rcParams.update({"font.size": 16})

idx = 2
run = training_runs[idx]

model = load_analysis_model(run)
