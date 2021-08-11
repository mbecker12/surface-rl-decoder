# Deep Q Learning

The Deep Q Learning program consists of multiple different subprocesses:
Multiple agent processes are spawned, each with their own local copy of the agent's neural network.
In one of those agent processes, multiple episodes are started and decoded by an ε-greedy policy based on the 
agent's current parameters.

Every step in each episode in each actor generates a transition tuple. These transition tuples are stored in a central
replay memory (side note: for the most part, prioritized experience replay is used in this project).

After a sufficient number of samples has been generated, the learner process starts learning (by stochastic gradient descent)
in a parallel process. This updates one central instance of the agent which is bound to the learner process. After a certain number of steps,
another, frozen target network which is used in the Deep Q Learning setup, is updated. Likewise, the learner process sends its new, learned and updated
weights to the actor processes, so that their agents (and thereby their respective policies) can be updated as well.

The actor process is defined in `actor.py`, the multiple environments per actor are bundled in an EnvironmentSet class defined in `environment_set.py`.
The replay memory buffer is defined in `io.py`, and the learner process is defined in `learner.py`.

There are many more utility packages in this directory which implement key functionality for the different subprocesses.

## Value Network Approach

The Value Network approach, on its phenomenological level, requires a new output structure of the chosen agent neural network.
Here, only one output neuron is required (or even expected). The network returns the state-value function V instead of the state-action-value function Q.
This brings with it some changes in the code.

In this project, `start_distributed_vnet.py` is responsible for starting the training process for Value Networks.
However, for the most part, it uses the same components as `start_distributed_mp.py` - the module used for Deep Q Learning.
To support both approaches, many modules used to spawn subprocesses contain if statements to distinguish the two cases and choose the correct action.
In fact, the biggest difference between the two entry point programs lies in the way the utility function which handles program configuration is executed. 