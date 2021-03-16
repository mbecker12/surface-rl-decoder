"""
Utility functions to choose and/or initialize the correct
learning model
(a.k.a. architecture)
(a.k.a. agent)
"""

import os
import torch
import yaml
from distributed.dummy_agent import DummyModel


def choose_model(model_name, model_config):
    """
    Given a model name, choose the corresponding neural network agent/model
    from a custom mapping

    Parameters
    ==========
    model_name: (str) valid name of the model/agent to be chosen
    model_config: (dict) dictionary containing expected model configuration.
        This may vary for different models

    Returns
    =======
    model: The desired neural network object, subclass of torch.nn.Module
    """

    if "dummy" in model_name:
        model = DummyModel(model_config)
    else:
        raise Exception(f"Error! Model '{model_name}' not supported or not recognized.")

    return model


def extend_model_config(
    model_config, system_size, stack_depth, num_actions_per_qubit=3
):
    """
    Extend an existing model or agent configuration dictionary
    with information about the environment.

    Parameters
    ==========
    model_config: (dict) dictionary contiaining information about
        model architecture and layer shapes
    system_size: (int) size of the state, ususally code distance+1
    stack_depth: (int) number of layers in a state stack
    num_actions_per_qubit: (optional) (int), number of possible actions on one
        qubit. Defaults to 3 for Pauli-X, -Y, -Z.

    Returns
    =======
    model_config: (dict) updated dictionary with configuration information
        of the model architecture
    """

    model_config["syndrome_size"] = system_size
    model_config["stack_depth"] = stack_depth
    model_config["num_actions_per_qubit"] = num_actions_per_qubit

    return model_config


def load_model(
    model: torch.nn.Module,
    old_model_path,
    load_criterion=False,
    optimizer=None):
    """
    Utility function to load a pytorch model's state dict from a specified path.

    Parameters
    ==========
    model: child class of torch.nn.Module, instance of neural network model
    old_model_path: path to save the state dict to
    load_criterion: (optional)(bool) whether to load the saved criterion
    optimizer: (optional)(toch.optim.Any) optimizer object; will be overwritten
        by saved state of the optimizer state_dict

    Returns
    =======
    model: model instance, overwritten with saved state in state_dict
    optimizer: optimizer instance, overwritten with saved state in state_dict
    criterion: loss instance, overwritten with saved state in state_dict
    """
    model.load_state_dict(torch.load(old_model_path))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(old_model_path + ".optimizer"))
    if load_criterion:
        criterion = torch.load(old_model_path + ".loss")
    else:
        criterion = None
    return model, optimizer, criterion


def save_model(model, optimizer, criterion, save_model_path):
    """
    Utility function to save a pytorch model's state dict.

    Parameters
    ==========
    model: child class of torch.nn.Module, instance of neural network model
    optimizer: optimizer object
    criterion: current loss
    save_model_path: path to save state_dicts to
    """
    head, _ = os.path.split(save_model_path)
    os.makedirs(head, exist_ok=True)

    torch.save(model.state_dict(), save_model_path)
    torch.save(optimizer.state_dict(), save_model_path + ".optimizer")
    torch.save(criterion, save_model_path + ".loss")


def save_metadata(config, path):
    """
    Save the metadata corresponding to a successful training run
    into a yaml file.
    Provides information for later analysis of training runs.

    Parameters
    ==========
    config: dictionary containing the configuration data of the training run
    path: path to store the metadata
    """
    head, _ = os.path.split(path)
    os.makedirs(head, exist_ok=True)

    with open(path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(config, yaml_file)
