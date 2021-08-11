# Overview

`actor_critic`: Code for PPO training. Contains entry point for PPO training (`start_ppo.py`).

`agents`: General collection of different Neural Network architectures

`analysis`: Scripts and utilities for post-training evaluation

`config`: Default configuration files for training

`distributed`: Modules for distributed training setup to run multiple instances on a computing cluster for increased efficiency. Contains the entry point for Q Learning (`start_distributed_mp.py`) and Value Network learning (`start_distributed_vnet.py`).

`evaluation`: Deprecated module with different manually-run scripts for evaluation and debugging purposes.

`surface_rl_decoder`: Defines the physical system of the surface code (with time component, i.e. stack depth) and the necessary utility functions.