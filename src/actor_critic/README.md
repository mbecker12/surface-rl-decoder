# PPO setup

This directory provides scripts and modules for PPO training.
Entry point is `start_ppo.py`.

Here, the general `PPO` class is initialized. Within this class, many more submodules are triggered:
Multiple multi-environment worker processes are started to step through different episodes in parallel. The driving force for choosing actions
is a central instance of one decoder agent. This creates multiple episode samples for the agent to train on; these samples are trained in a central replay buffer. The agent is trained once sufficiently many samples are stored in the replay buffer. The training is done on multiple random draws of these samples before the replay buffer is cleared and filled anew.