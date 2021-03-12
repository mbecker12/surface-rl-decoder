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

You can leave the environment by executing

    deactivate


The Project was setup using PyScaffold; after setting up your environment, you should run

    python setup.py develop

to set up the project for development.


Configurations
==============

This project uses the config-env-parser to define configuration parameters of all kinds.
This package will look for a .ini file in the src directory and extract the parameters from there.
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

Deployment
==========

An example of how to deploy a job script:

    sbatch surface-rl-decoder/alvis-job-first.sh --export-file=surface-rl-decoder/conf.env 

The actual job script is `alvis-job-first.sh`.

Since we make use of the config-ini-parser, we can override the settings with environment variables which
we specify in the `conf.env` in the above example.

Build
=====

We can build and push a docker image based on the `Dockerfile` in this repository.

The job script mentioned above then envokes `singularity` to create and run a singularity image based on said docker image on the cluster.

Tests
=====

Unit tests are executed in the CI pipeline (under the section "Actions" in github)
or can be run locally.

You first need to install the test requirements:

    pip install -r test-requirements.txt

Then, the tests including coverage report can be run via

    python -m pytest --cov-report=html --cov=src

The detailed coverage report can be obtained in ./htmlcov/index.html.


Note
====

This project has been set up using PyScaffold 3.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
