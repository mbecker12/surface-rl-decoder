
import torch
from agents import Conv_2d_agent
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import vector_to_parameters, clip_grad_norm
from distributed.environment_set import EnvironmentSet
from distributed.model_util import choose_model, load_model
import torch.optim as optim
import gym
import logging
import random
import math

class DQN_Agent:
    """Interacts with and learns from the environment"""

    def __init__(self, config):


        """
        Create agent network for the hindsight learning. 
        Uses a config dictionary which holds keys bound 
        to neccessary parameters. The neccessary ones are:

        Parameters
        ======
            network: (str) dqn network type ......see if this can be implemented with eval or similar
            batch_size: (int) size of batches
            buffer_size: (int) size of replay memory
            learning_rate: (float) learning rate of the algorithm
            tau: (float) tau for soft updating th network weights
            gamma: (float) discount factor
            device: (str) device that is used for computation
            seed: (int) random seed
            update_every: (int) update frequency
            bit_length: (int) length of bit map #will need to be revised to properly extract state
            n_step: (int) n_step for how far one should consider the rewards
            other parameters required for the neural network (dependant on the network)

        """
        seed = int(config.get("seed",-1))
        self.device = config.get("device")
        self.tau = float(config.get("tau"))
        self.gamma = float(config.get("gamma"))
        self.update_every = int(config.get("update_every"))
        self.batch_size = int(config.get("batch_size"))
        self.buffer_size = int(config.get("buffer_size"))
        learning_rate = float(config.get("learning_rate"))
        load_model_flag = bool(config.get("load_model_flag"))
        self.bit_length = int(config.get("bit_length")) #This may need to be revised
        self.model_name = str(config.get("model_name"))
        self.q_updates = 0
        self.n_step = int(config.get("n_step", 0))
        self.current_episode = []
        self.action_step = 4
        self.last_action = None



        if seed != -1:
            random.seed(seed)

        #Q-network

        self.qnetwork_local = choose_model(self.model_name, config)
        self.qnetwork_target = choose_model(self.model_name, config)
        if load_model_flag:
            old_model_path = config.get("old_model_path")
            self.qnetwork_local, _, _ = load_model(self.qnetwork_local, old_model_path) 
            self.qnetwork_target, _, _ = load_model(self.qnetwork_target, old_model_path)
            logger.info(f"Loaded actor model from {old_model_path}")

        self.qnetwork_local.to(device)
        self.qnetwork_target.to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = learning_rate)
        print(self.qnetwork_local)

        #Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device, seed, self.gamma, self.n_step)

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
                    #logger.info("--- Her Sampling ---")
                    #logger.info("new goal: {}".format(new_goal))

                    r_ = self.reward_function(state[:self.bit_length], new_goal) #next_state #will need to be revised due to changed structure
                    #logger.info("new reward: {}".format(r_))

                    next_state = np.concatenate((next_state[:self.bit_length], new_goal)) #will need to be revised to properly extract state
                    state = np.concatenate((state[:self.bit_length], new_goal)) #will need to be revised to properly extract state

                    if(next_state[:self.bit_length] == new_goal).all(): #will need to be revised to properly extract state
                        d = 1
                    else:
                        d = 0
                    self.memory.add(state, action, r_, next_state, d)

            N = len(self.current_episode)
            for _in range(N):
                #If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    loss = self.learn(experiences)
                    self.Q_updates += 1
                    writer.add_scalar("Q_loss", loss, self. Q_updates)
            self.current_episode = []

    def reward_function(self, state, goal):
        return if (state == goal).all() else -1

    def sample_goals(self, idx, n):
        new_goals = []
        for _in range(n):
            transition = random.choice(self.current_episode[idx:])

            new_goal = transition[0][:self.bit_length] #will need to be revised to properly extract state
            new_goals.append(new_goal)
        return new_goals

    def act(self, state, eps = 0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames

        Parameters
        ==========
            frame: to adjust epsilon
            state (array_like): current state

        """

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used
            action = np.argmax(action_values.cpu().data.numpy())
            self.last_action = action
            return action


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples

        Parameters
        ==========
            experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples
            gamma: (float) discount factor
        """

        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        #Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Compute Q targets for current states
        Q_targets = rewards + (self.gamma**self.n_step * Q_targets_next * (1-dones))
        #get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1,actions)
        #Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) #mse_loss
        #Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        #-------------update target network --------------#
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()


    def soft_update(self, local_model, target_model):
        """Soft update model parameters
        theta_target = tau*theta_local + (1-tau)*theta_target
        Parameters
        ==========
            local_model: (PyTorch model) weights will be copied from
            target_model: (PyTorch model)
            tau: (float) interpolation parameter

        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0-self.tau) * target_param.data)

