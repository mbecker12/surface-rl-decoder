==================
surface-rl-decoder
==================

Train agents via reinforcement learning for decoding on qubit surface codes 


Description
===========

To make quantum computation more error-safe, one usually encodes one logical
qubit with multiple physical qubits.

One such realization is the surface code, which is the base for our work.
On the surface code, the different qubits can still be subject to physical errors
(e.g. bit flip or phase flip).
The errors themselves can't be measured as it would destroy the quantum nature of
the system. Instead, local parity checks - so-called syndrome measurements - are performed.
Those can however be subject to noise themselves, which introduces measurement errors.
This in turn makes the problem non-Markovian as the latest state isn't representative
the whole time evolution of the system anymore.

Here, we set up a new environment to reflect the time evolution of syndrome measurements.
Then, our goal is to train agents to be able to decode the erroneous surface code
such that the encoded logical qubit is restored and can potentially be used for further
quantum computations.

Code Environment
================

The code was written in Python 3.8.5.

A virtual environment for this project can be setup via

    python3 -m venv ~/virtualenv/qec
    
    source ~/virtualenv/qec/bin/activate
    
    pip install -r requirements.txt

You can leave the environment by executing ``deactivate``.


The Project was setup using PyScaffold; after setting up your environment, you should run

    python setup.py develop

to set up the project for development.


Deployment
==========

An example of how to deploy a job script:

    sbatch Example: mp-script.sh -c conf.env -i qec-mp.sif -t runs -T tmp_runs -n networks -N tmp_networks -w surface-rl-decoder -d "describe the purpose of the current run"

The actual job script is ``mp-script.sh`` and supports the functionality to adjust paths and configurations in the ``singularity`` container
(see below for more details on how to build a singularity image).

Inside the job script, a ``conf.env`` file is mounted into the container.
Since we make use of the config-ini-parser, we can override the settings with environment variables which
we specify in the environment file. Custom configuration should be done by changing the content of the ``conf.env`` file.

The job script accepts multiple shorthand flags with arguments. See ``bash mp-script.sh -h`` for usage help.
The input arguments allow you for example to define the path to an environment-config file. This can be used to specify an environment file
to change the configuration of the run and specify things like system size, network architecture, path to network configuration, etc.
The configuration variables are described in a later section.

Logging
=======

By default, logging of informative metrics happens via tensorboard. The logging and monitoring happens in different verbosity levels,
which can be set for the different processes in the configuration (or, again, as environment variables).
Besides that, there is the possibility to output performance measures of certain code blocks to `stdout` via the logger.
The performance measurements are dependent on the `benchmarking` parameter and partially on the `verbosity` parameter as well.
Below is an overview of which monitoring metrics and performance measures are being tracked at which verbosity level.

+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| Verbosity Level | Logged Metrics                                                                                                                                | Logger Level | Performance Measure (Requires ``benchmarking`` Parameter)                                                  | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| 1               | Learner: Evaluation; GPU: Memory Info; IO: Total, Speed, Sample Rewards from Actor                                                            | INFO         | Learner: Q Learning Step, Evaluation; IO: Save Actor Data; Actor: Select Action, Step through Environments | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| 2               | Prioritized Experience Replay: beta; Actor: epsilon, effect of intermediate reward                                                            |              |                                                                                                            | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| 3               | IO: Sample Actions from Actor; CPU: Memory Info; Learner: Received Data                                                                       | DEBUG        | Learner: Update Target Network; IO: Send Data to Learner, Priority Update                                  | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| 4               | Actor: Sent Data; PER: Sampling Metrics (Priorities, Weights, Indices), NN: First Layer, Last Layer, IO: Sent Priorities, Received Priorities |              |                                                                                                            | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+
| 5               | Visualized Transition Samples: State, Next State, Action Grid                                                                                 |              |                                                                                                            | 
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+--------------+------------------------------------------------------------------------------------------------------------+



Configurations
==============

This project uses the config-env-parser to define configuration parameters of all kinds.
This package will look for a ``.ini`` file in the ``src`` directory (and its subdirectories) and extract the parameters from there.
If a parameter exists as an environment variable, the environment variable has higher priority
and its value will be used.

+---------------------------------------+--------------------------+---------------+
| Configuration dict from .ini file     | Environment variable     | Default value |
+=======================================+==========================+===============+
| cfg["config"]["env"]["size"]          | CONFIG_ENV_SIZE          | 5             |
+---------------------------------------+--------------------------+---------------+
| cfg["config"]["env"]["min_qbit_err"]  | CONFIG_ENV_MIN_QBIT_ERR  | 0             |
+---------------------------------------+--------------------------+---------------+
| cfg["config"]["env"]["p_error"]       | CONFIG_ENV_P_ERROR       | 0.1           |
+---------------------------------------+--------------------------+---------------+
| cfg["config"]["env"]["p_msmt"]        | CONFIG_ENV_P_MSMT        | 0.05          |
+---------------------------------------+--------------------------+---------------+
| cfg["config"]["env"]["stack_depth"]   | CONFIG_ENV_STACK_DEPTH   | 8             |
+---------------------------------------+--------------------------+---------------+
| cfg["config"]["env"]["error_channel"] | CONFIG_ENV_ERROR_CHANNEL | "dp"          |
+---------------------------------------+--------------------------+---------------+

Note: The config variable `CONFIG_GENERAL_SUMMARY_DATE` for setting the subdirectory
to save tensorboard info and networks, cannot be overwritten by conf.env files.

It can however be overwritten by the `-s` flag of the job script `mp-script.sh`.
This use is intended for testing purposes to keep overwriting the same testing
directory, e.g. `networks/test` and `runs/test`.

Build
=====

We can build and push a docker image based on the ``Dockerfile`` in this repository.
For execution on an HPC cluster, the docker imager should be transformed to a ``singularity`` image.
This can be done by running

    singularity build $singularity_image_name $location_of_docker_image

The job script mentioned above then envokes ``singularity`` to load a singularity image based on said docker image on the cluster.


Tests
=====

Unit tests are executed in the CI pipeline (under the section "Actions" in github)
or can be run locally.

You first need to install the test requirements:

    pip install -r test-requirements.txt

Then, the tests including coverage report can be run via

    python -m pytest --cov-report=html --cov=src

The detailed coverage report can be obtained in ``./htmlcov/index.html.``


Note
====

This project has been set up using PyScaffold 3.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
