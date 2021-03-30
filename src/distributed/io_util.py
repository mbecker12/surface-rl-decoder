"""
A collection of very low-key utility functions for the io,
aka replay-memory module
"""
import psutil
import nvgpu
import numpy as np


def assert_transition_shapes(transition, stack_depth, syndrome_size):
    """
    Assert that the incoming transitions conform to the expected array shape.

    transition: incoming transition to check
    stack_depth: syndrome state stack depth, umber of layers in stack
    syndrome_size: number of rows/columns in the syndrome state
    """
    ## if the zip method is chosen in the actor
    assert transition[0].shape == (
        stack_depth,
        syndrome_size,
        syndrome_size,
    ), transition[0].shape
    assert transition[1].shape == (3,), transition[1].shape
    assert isinstance(transition[2], (float, np.float64, np.float32)), type(
        transition[2]
    )
    assert transition[3].shape == (
        stack_depth,
        syndrome_size,
        syndrome_size,
    ), transition[3].shape
    assert isinstance(transition[4], (bool, np.bool_)), type(transition[4])


def add_transition_images_to_tensorboard(
    tensorboard, transition, tensorboard_step, terminal_action, current_time_tb
):
    """
    Generate little images from state, next_state, and action and send them to tensorboard.
    To have a quick glance if the action and step functionality in the environment
    works properly.
    Should probably only be run at a very high verbosity level.
    """
    transition_shape = transition[0][-1].shape

    _state_float = transition[0][-1].astype(np.float32)
    _next_state_float = transition[3][-1].astype(np.float32)

    tensorboard.add_image(
        "transition/state",
        _state_float,
        tensorboard_step,
        dataformats="HW",
        walltime=current_time_tb,
    )
    tensorboard.add_image(
        "transition/next_state",
        _next_state_float,
        tensorboard_step,
        dataformats="HW",
        walltime=current_time_tb,
    )
    action_matrix = np.zeros(
        (transition_shape[0] - 1, transition_shape[1] - 1),
        dtype=np.float32,
    )
    action = transition[1]
    action_matrix[action[0], action[1]] = action[-1] / max(terminal_action, 3)
    tensorboard.add_image(
        "transition/action_viz",
        action_matrix,
        tensorboard_step,
        dataformats="HW",
        walltime=current_time_tb,
    )


def monitor_gpu_memory(tensorboard, current_time, performance_start, current_time_tb):
    """
    Access and output the status of the VRAM of the GPU to tensorboard.
    """
    gpu_info = nvgpu.gpu_info()
    for i in gpu_info:
        gpu_info = "io/{} {}".format(i["type"], i["index"])
        gpu_mem_total = i["mem_total"]
        gpu_mem_used = i["mem_used"]
        tensorboard.add_scalars(
            gpu_info,
            {
                "gpu_mem_total": gpu_mem_total,
                "gpu_mem_used": gpu_mem_used,
            },
            current_time - performance_start,
            walltime=current_time_tb,
        )


def monitor_cpu_memory(tensorboard, current_time, performance_start, current_time_tb):
    """
    Output the status of RAM.
    """
    mem_usage = psutil.virtual_memory()
    memory_total = mem_usage.total >> 20
    mem_available = mem_usage.available >> 20
    mem_used = mem_usage.used >> 20
    tensorboard.add_scalars(
        "io/cpu",
        {
            "mem_total / MB": memory_total,
            "mem_available / MB": mem_available,
            "mem_used / MB": mem_used,
        },
        current_time - performance_start,
        walltime=current_time_tb,
    )


def monitor_data_io(
    tensorboard,
    data_consumption_total,
    transitions_total,
    count_consumption_outgoing,
    count_transition_received,
    stop_watch,
    current_time,
    performance_start_time,
    current_time_tb,
):
    """
    Utility function to log data input/output via tensorboard.
    """
    tensorboard.add_scalars(
        "io/total",
        {
            "total batch consumption outgoing": data_consumption_total,
            "total # received transitions": transitions_total,
        },
        current_time - performance_start_time,
        walltime=current_time_tb,
    )
    tensorboard.add_scalars(
        "io/speed",
        {
            "batch consumption rate of outgoing transitions": count_consumption_outgoing
            / (current_time - stop_watch),
            "received transitions rate": count_transition_received
            / (current_time - stop_watch),
        },
        current_time - performance_start_time,
        walltime=current_time_tb,
    )


def handle_transition_monitoring(
    tensorboard,
    transition,
    verbosity,
    tensorboard_step,
    current_time_tb,
    terminal_action,
):
    """
    Output information about a transition set, made up of
    (state, action, reward, next_state, is_terminal)
    """
    tensorboard.add_scalars(
        "transition/reward",
        {"reward": transition[2]},
        tensorboard_step,
        walltime=current_time_tb,
    )

    if verbosity >= 3:
        tensorboard.add_scalars(
            "transition/action",
            {
                "x": transition[1][0],
                "y": transition[1][1],
                "action": transition[1][2],
            },
            tensorboard_step,
            walltime=current_time_tb,
        )

    if verbosity >= 5:
        add_transition_images_to_tensorboard(
            tensorboard,
            transition,
            tensorboard_step,
            terminal_action,
            current_time_tb,
        )
