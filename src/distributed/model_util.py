"""
Utility functions to choose and/or initialize the correct
learning model
(a.k.a. architecture)
(a.k.a. agent)
"""

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
