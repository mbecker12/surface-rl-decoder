# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for surface_rl_decoder.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import os
import json
import pytest
import numpy as np
from src.distributed.model_util import choose_model, extend_model_config
from src.surface_rl_decoder.surface_code import SurfaceCode
from src.distributed.environment_set import EnvironmentSet


@pytest.fixture
def sc():
    return SurfaceCode()


@pytest.fixture
def qbs():
    return np.zeros(shape=(8, 5, 5), dtype=np.uint8)


@pytest.fixture
def configure_env():
    def configure_env_inner():
        original_depth = os.environ.get("CONFIG_ENV_STACK_DEPTH", "4")
        os.environ["CONFIG_ENV_STACK_DEPTH"] = "4"
        original_size = os.environ.get("CONFIG_ENV_SIZE", "5")
        os.environ["CONFIG_ENV_SIZE"] = "5"
        original_error_channel = os.environ.get("CONFIG_ENV_ERROR_CHANNEL", "x")
        os.environ["CONFIG_ENV_ERROR_CHANNEL"] = "x"

        return original_depth, original_size, original_error_channel

    return configure_env_inner


@pytest.fixture
def restore_env():
    def restore_env_inner(original_depth, original_size, original_error_channel):
        os.environ["CONFIG_ENV_STACK_DEPTH"] = original_depth
        os.environ["CONFIG_ENV_SIZE"] = original_size
        os.environ["CONFIG_ENV_ERROR_CHANNEL"] = original_error_channel

    return restore_env_inner


@pytest.fixture
def seed_surface_code():
    def seed_sc(surface_code, seed, p_error, p_msmt, error_channel):
        np.random.seed(seed)
        surface_code.p_error = p_error
        surface_code.p_msmt = p_msmt
        surface_code.reset(error_channel=error_channel)
        return surface_code

    return seed_sc


@pytest.fixture
def env_set():
    return EnvironmentSet(SurfaceCode(), 5)


@pytest.fixture
def init_testing_model():
    def testing_model(args, state_size, stack_depth):
        model_name = args["model_name"]
        model_config = args["model_config"]
        model_config = extend_model_config(model_config, state_size, stack_depth)
        model = choose_model(model_name, model_config)
        return model

    return testing_model


MODEL_CONFIG_LOCATION = "src/config/model_spec"


@pytest.fixture
def load_model_config():
    def get_config_json(model_config_file, model_name):
        model_config_file_path = os.path.join(MODEL_CONFIG_LOCATION, model_config_file)
        with open(model_config_file_path) as json_file:
            model_config = json.load(json_file)

        # select the specification of the right model from the json
        model_config = model_config[model_name]

        return model_config

    return get_config_json
