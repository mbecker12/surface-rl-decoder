import os
from time import time
import torch

# pylint: disable=no-member
from src.agents.dummy_agent import DummyModel
from src.evaluation.batch_evaluation import batch_evaluation


def test_batch_evaluation(configure_env, restore_env):
    original_depth, original_size, original_error_channel = configure_env()
    config_ = {
        "layer1_size": 512,
        "layer2_size": 512,
        "code_size": 5,
        "syndrome_size": int(os.environ.get("CONFIG_ENV_SIZE")) + 1,
        "stack_depth": int(os.environ.get("CONFIG_ENV_STACK_DEPTH")),
        "num_actions_per_qubit": 3,
        "device": "cpu"
    }
    model_ = DummyModel(config=config_)

    device_ = torch.device("cpu")

    start_time = time()
    for _ in range(3):
        eval_metrics = batch_evaluation(model_, "", device_, p_err=0.01, p_msmt=0.0)

        print(f"{eval_metrics=}")

    end_time = time()
    print(f"Time elapsed: {end_time - start_time}")
    restore_env(original_depth, original_size, original_error_channel)
