import pandas as pd
from dataclasses import dataclass

@dataclass
class TrainingRun():
    job_id: int
    code_size: int
    stack_depth: int
    p_err: float
    p_msmt: float
    rl_type: str
    architecture: str
    data: pd.DataFrame = None
    duration: float = None
    model_name: str = None
    model_config_file: str = "conv_agents_slim.json"
    transfer_learning: int = 0