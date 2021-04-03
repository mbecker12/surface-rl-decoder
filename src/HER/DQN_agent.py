
import torch
from agents import Conv_2d_agent
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import vector_to_parameters, clip_grad_norm
import torch.optim as optim
import gym
import logging
import random
import math

class DQN_Agent():
    """Interacts with and learns from the environment"""

    def __init__(self, config):


        """
        Create agent network for the hindsight learning. 
        Uses a config dictionary which holds keys bound 
        to neccessary parameters. The neccessary ones are:

        Parameters
        ======
            code_size: (int) size of the code
            num_actions_per_qubit: (int) in most cases should be 3 but can be adjusted,
                it is the number of types of correction actions that can be applied on the qubit
            network: (str) dqn network type
            batch_size: (int) size of batches
            buffer_size: (int) size of replay memory
            learning_rate: (float) learning rate of the algorithm
            tau: (float) tau for soft updating th network weights
            gamma: (float) discount factor
            device: (str) device that is used for computation
            seed: (int) random seed
            update_every: (int) update frequency
            n_step: (int) n_step
            other parameters required for the neural network (dependant on the network)

        """"
        self.code_size = int(config.get("code_size"))
        self.num_actions_per_qubit = int(config.get("num_actions_per_qubit"))
        self.seed = int(config.get("seed"))
        self.device = config.get("device")
        self.tau = float(config.get("tau"))
        self.gamma = float(config.get("gamma"))
        self.update_every = int(config.get("update_every"))
        self.batch_size = int(config.get("batch_size"))
        self.buffer_size = int(config.get("buffer_size"))
        self.learning_rate = float(config.get("learning_rate"))
        self.q_updates = 0
        self.n_step = int(config.get("n_step"))
        self.current_episode = []
        self.action_step = 4
        self.last_action = None

        #Q-network

        self.qnetwork_local = Conv_2d_agent(config).to(self.device) #see if we can make this choosable via config input, otherwise make it an explicit input
        self.qnetwork_target = Conv_2d_agent(config).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)
        print(self.qnetwork_local)

        #Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device, self.seed, self.gamma, self.n_step)

        #Initialize time step for the update steps
        self.t_step = 0


    def step(self, state, action, reward, next_state, done, writer, current_goal):

        self.current_episode.append([state, action, reward, next_state, done])
        if done == 1:
            for idx, exp in enumerate(self.current_episode):
                state, action, reward, next_state, done = exp
                #logger.info("Episode state: {}".format(state))
                #logger.info("Episode action: {}".format(action))
                #logger.info("Episode reward: {}".format(reward))
                #logger.info("Episode next_state: {}".format(next_state))

                #Save experience in replay memory
                self.memory.add(state, action, reward, next_state, done)

                #Sample additional goals for HER
                new_goals = self.sample_goals(idx, 4)
                #logger.info("new goal: {}".format(new_goal))

                for new_goal in new_goals: