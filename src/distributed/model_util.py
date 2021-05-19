"""
Utility functions to choose and/or initialize the correct
learning model
(a.k.a. architecture)
(a.k.a. agent)
"""

import os
from typing import Tuple, Union
from numpy import add
import torch
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam
from torch import nn
import yaml
import json
from agents.base_agent import BaseAgent
from agents.conv_2d_agent import Conv2dAgent
from agents.conv_3d_agent import Conv3dAgent
from agents.old_conv_3d_agent import Conv3dAgent as OldConv3dAgent
from agents.old_conv_2d_agent import SimpleConv2D as OldConv2dAgent
from agents.dummy_agent import DummyModel


def choose_old_model(model_name, model_config):
    if "dummy" in model_name:
        model = DummyModel(model_config)
    elif "conv2d" in model_name.lower():
        model = Conv2dAgent(model_config)
    elif "conv3d" in model_name.lower():
        model = OldConv3dAgent(model_config)
    else:
        raise Exception(f"Error! Model '{model_name}' not supported or not recognized.")

    return model


def choose_model(
    model_name,
    model_config,
    model_config_base=None,
    model_path_base=None,
    transfer_learning=1,
):
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
    elif model_name.lower() == "conv2d":
        if transfer_learning:
            assert (
                model_path_base is not None
            ), "Need to specify the path of the pretrained base model!"
            assert (
                model_config_base is not None
            ), "Need to specify a configuration file for the base model!"
            model = configure_pretrained_model(
                model_config_base,
                model_config,
                model_path_base,
                model_class_base=OldConv2dAgent,
                model_class_top=Conv2dAgent,
            )
            print("Prepare Conv2dAgent and OldConv2dAgent using transfer learning.")
        else:
            model = Conv2dAgent(model_config)
            print("Prepare Conv2dAgent w/o transfer learning")
    elif "conv3d" in model_name.lower():
        model = Conv3dAgent(model_config)
    else:
        raise Exception(f"Error! Model '{model_name}' not supported or not recognized.")

    return model


def extend_model_config(
    model_config, syndrome_size, stack_depth, num_actions_per_qubit=3, device="cpu"
):
    """
    Extend an existing model or agent configuration dictionary
    with information about the environment.

    Parameters
    ==========
    model_config: (dict) dictionary contiaining information about
        model architecture and layer shapes
    syndrome_size: (int) size of the state, ususally code distance+1
    stack_depth: (int) number of layers in a state stack
    num_actions_per_qubit: (optional) (int), number of possible actions on one
        qubit. Defaults to 3 for Pauli-X, -Y, -Z.

    Returns
    =======
    model_config: (dict) updated dictionary with configuration information
        of the model architecture
    """

    model_config["syndrome_size"] = syndrome_size
    model_config["code_size"] = syndrome_size - 1
    model_config["stack_depth"] = stack_depth
    model_config["num_actions_per_qubit"] = num_actions_per_qubit
    model_config["device"] = device

    return model_config


def load_model(
    model: torch.nn.Module,
    old_model_path: str,
    load_criterion=False,
    load_optimizer=False,
    learning_rate=None,
    optimizer_device=None,
    model_device=None,
) -> Tuple[nn.Module, Union[Adam, None], Union[MSELoss, None]]:
    """
    Utility function to load a pytorch model's state dict from a specified path.

    Parameters
    ==========
    model: child class of torch.nn.Module, instance of neural network model
    old_model_path: path to save the state dict to
    model_device: (optional) device for the loaded model
    load_criterion: (optional)(bool) whether to load the saved criterion
    load_optimizer: (optional)(bool) whether to load the saved optimizer
    optimizer_device: (optional, required if load_optimizer) device for the loaded optimizer
    learning_rate: (optional, required if load_optimizer) learning rate for gradient descent

    Returns
    =======
    model: model instance, overwritten with saved state in state_dict
    optimizer: optimizer instance, overwritten with saved state in state_dict
    criterion: loss instance, overwritten with saved state in state_dict
    """
    # load model
    model.load_state_dict(torch.load(old_model_path, map_location=model_device))
    if model_device is not None:
        model = model.to(model_device)

    # load optimizer
    if load_optimizer:
        assert learning_rate is not None
        optimizer = Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(
            torch.load(old_model_path + ".optimizer", map_location=optimizer_device)
        )
        assert optimizer_device is not None
        optimizer = optimizer_to(optimizer, optimizer_device)
    else:
        optimizer = None

    # load criterion
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


def optimizer_to(optim, device):
    """
    Send a torch.optim object to the target device.
    """
    # you gotta do what you gotta do...
    # pylint: disable=protected-access
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

    return optim


def configure_pretrained_model(
    model_config_base,
    model_config_top,
    model_path_base,
    device_str="cpu",
    model_class_base: Union[BaseAgent, OldConv2dAgent, OldConv3dAgent] = None,
    model_class_top: Union[BaseAgent, Conv2dAgent, Conv3dAgent] = None,
    model_name_base: str = None,
    model_name_top: str = None,
):
    """
    Load a pretrained model base and combine it with a
    untrained model top.
    This assumes the model base to be a convolutional backbone
    and the model top to contain a recurrent unit (RNN, GRU, LSTM, or GTrXL)
    plus linear layers.
    """

    if model_class_base is not None:
        # sorry for hard-coding this but loading the old model only works on this parameter
        model_config_base["stack_depth"] = 4
        model_config_base["code_size"] = 5
        model_base = model_class_base(model_config_base)

    elif model_name_base is not None:
        if model_name_base == "old_conv_2d":
            # sorry for hard-coding this but loading the old model only works on this parameter
            model_config_base["stack_depth"] = 4
            model_config_base["code_size"] = 5
            model_base = OldConv2dAgent(model_config_base)
        elif model_name_base == "conv_2d":
            # do other stuff here
            pass
        else:
            raise Exception(
                f"Error! Model name {model_name_base} is not supported for the network base."
            )
    else:
        raise Exception(
            f"Error! You need to provide either a class or an identifier string for the model base."
        )

    # reconfigure channels
    # assume + hardcode that old model has 4 conv layers
    out_channels = model_base.comp_conv_layer_both.weight.shape[0]

    model_config_top["channel_list"] = [
        model_base.input_conv_layer_both.out_channels,
        model_base.nd_conv_layer_both.out_channels,
        model_base.rd_conv_layer_both.out_channels,
        model_base.comp_conv_layer_both.out_channels,
    ]

    if model_class_top is not None:
        model_top = model_class_top(model_config_top)

    elif model_name_top is not None:
        if model_name_top == "conv_2d":
            model_top = Conv2dAgent(model_config_top)
        else:
            raise Exception(
                f"Error! Model name {model_name_top} is not supported for the network head."
            )
    else:
        raise Exception(
            "Error! You need to provide either a class or an identifier string for the model top."
        )

    device = model_config_top["device"]

    model_base.load_state_dict(
        torch.load(model_path_base, map_location=torch.device(device))
    )

    # need to hardcode the number of available layers for now
    # since it was default in the old 2D Conv model configuration

    model_top.conv1 = model_base.input_conv_layer_both
    model_top.conv2 = model_base.nd_conv_layer_both
    model_top.conv3 = model_base.rd_conv_layer_both
    model_top.conv4 = model_base.comp_conv_layer_both

    model_top.conv1.requires_grad = False
    for param in model_top.conv1.parameters():
        param.requires_grad = False

    model_top.conv2.requires_grad = False
    for param in model_top.conv2.parameters():
        param.requires_grad = False

    model_top.conv3.requires_grad = False
    for param in model_top.conv3.parameters():
        param.requires_grad = False

    model_top.conv4.requires_grad = False
    for param in model_top.conv4.parameters():
        param.requires_grad = False

    return model_top


if __name__ == "__main__":
    model_name_base = "old_conv_2d"
    model_name_top = "conv_2d"
    model_config_path_base = "src/config/model_spec/old_conv_agents.json"
    model_config_path_top = "src/config/model_spec/conv_agents_slim.json"
    model_path_base = "remote_networks/5/65280/simple_conv_5_65280.pt"

    def add_model_size(config, config_file_path):
        if "slim" in config_file_path:
            config["network_size"] = "slim"
        elif "large" in config_file_path:
            config["network_size"] = "large"
        else:
            config["network_size"] = "medium"
        return config

    with open(model_config_path_base, "r") as json_file:
        model_config_base = json.load(json_file)

    model_config_base = model_config_base["simple_conv"]

    code_size = 5
    syndrome_size = code_size + 1
    stack_depth = 4
    model_config_base = extend_model_config(
        model_config_base, syndrome_size, stack_depth
    )
    model_config_base["device"] = "cpu"

    with open(model_config_path_top, "r") as json_file_top:
        model_config_top = json.load(json_file_top)

    model_config_top = model_config_top["conv2d"]

    stack_depth = 5
    syndrome_size = code_size + 1
    model_config_top = extend_model_config(model_config_top, syndrome_size, stack_depth)
    model_config_top["device"] = "cpu"

    model_config_top = add_model_size(model_config_top, model_config_path_top)

    frankensteins_model = configure_pretrained_model(
        model_config_base,
        model_config_top,
        model_path_base,
        device_str="cpu",
        model_class_base=OldConv2dAgent,
        model_class_top=Conv2dAgent,
    )
    print(f"{frankensteins_model.stack_depth=}")
    print(f"{frankensteins_model.size=}")

    for name, param in frankensteins_model.named_parameters():
        print(f"{name=}: grad? {param.requires_grad}! Shape: {param.shape}")

    import numpy as np

    batch_size = 8
    states = np.zeros((batch_size, stack_depth, code_size + 1, code_size + 1))
    torch_states = torch.tensor(
        states, device=torch.device(model_config_top["device"]), dtype=torch.float32
    )

    output = frankensteins_model(torch_states)
    assert output.shape == (batch_size, 3 * code_size * code_size + 1)
    print(f"{output.shape=}")

    import matplotlib.pyplot as plt

    if True:
        layer_one = frankensteins_model.conv1.weight.detach().cpu().numpy()
        channels, _, x, y = layer_one.shape

        stacked_weights = np.hstack(
            [
                np.pad(
                    layer_one[i, 0], ((0, 0), (0, 1)), "constant", constant_values=-2
                )
                for i in range(channels)
            ]
        )

        fig, ax = plt.subplots(3, 1)

        ax[0].imshow(stacked_weights, cmap="Greys", vmin=-0.8, vmax=0.3)
        ax[0].set(title="Pretrained Weights")

        random_model = Conv2dAgent(model_config_top)
        random_layer_one = random_model.conv1.weight.detach().cpu().numpy()
        random_channels, _, random_x, random_y = random_layer_one.shape

        random_stacked_weights = np.hstack(
            [
                np.pad(
                    random_layer_one[i, 0],
                    ((0, 0), (0, 1)),
                    "constant",
                    constant_values=-1,
                )
                for i in range(random_channels)
            ]
        )
        ax[1].set(title="Random Weights")
        ax[1].imshow(random_stacked_weights, cmap="Greys", vmin=-0.8, vmax=0.3)

        stacked_weights_dist = stacked_weights[stacked_weights > -1].flatten()
        weights_hist = np.histogram(stacked_weights_dist, bins=10)

        random_stacked_weights_dist = random_stacked_weights[
            random_stacked_weights > -1
        ].flatten()
        random_weights_hist = np.histogram(random_stacked_weights_dist, bins=10)

        ax[2].plot(weights_hist[1][:-1], weights_hist[0], label="pretrained")
        ax[2].plot(random_weights_hist[1][:-1], random_weights_hist[0], label="random")
        ax[2].set(title="Weights Distribution")
        plt.legend()
        plt.show()

        layer_two = frankensteins_model.conv2.weight.detach().cpu().numpy()
        channels, _, x, y = layer_two.shape

        stacked_weights = np.hstack(
            [
                np.pad(
                    layer_two[i, 0], ((0, 0), (0, 1)), "constant", constant_values=-2
                )
                for i in range(channels)
            ]
        )

        fig, ax = plt.subplots(3, 1)

        ax[0].imshow(stacked_weights, cmap="Greys", vmin=-0.8, vmax=0.3)
        ax[0].set(title="Pretrained Weights")

        random_model = Conv2dAgent(model_config_top)
        random_layer_two = random_model.conv2.weight.detach().cpu().numpy()
        random_channels, _, random_x, random_y = random_layer_two.shape

        random_stacked_weights = np.hstack(
            [
                np.pad(
                    random_layer_two[i, 0],
                    ((0, 0), (0, 1)),
                    "constant",
                    constant_values=-1,
                )
                for i in range(random_channels)
            ]
        )
        ax[1].set(title="Random Weights")
        ax[1].imshow(random_stacked_weights, cmap="Greys", vmin=-0.8, vmax=0.3)

        stacked_weights_dist = stacked_weights[stacked_weights > -1].flatten()
        weights_hist = np.histogram(stacked_weights_dist, bins=10)

        random_stacked_weights_dist = random_stacked_weights[
            random_stacked_weights > -1
        ].flatten()
        random_weights_hist = np.histogram(random_stacked_weights_dist, bins=10)

        ax[2].plot(weights_hist[1][:-1], weights_hist[0], label="pretrained")
        ax[2].plot(random_weights_hist[1][:-1], random_weights_hist[0], label="random")
        ax[2].set(title="Weights Distribution")
        plt.legend()
        plt.show()
