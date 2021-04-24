""" Fixed-size buffer to store experience tuples. """

from collections import namedtuple, deque
import random
import torch
import numpy as np


#may want to readjust to be able to take multiple environments
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device, gamma, n_step = 1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: (int) maximum size of buffer
            batch_size: (int) size of each training batch
            seed: (int) random seed
            n_step: (int) steps for the single step buffer before it creates an experience of n_steps
                used if one desires a multi-step experience
        """

        self.device = device
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done", "goal"])
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen = self.n_step)


    def add(self, state, action, reward, next_state, done, goal):
        """Add a new experience to memory"""
        #print("before:", state, action, reward, next_state, done, goal)
        self.n_step_buffer.append((state, action, reward, next_state, done, goal))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            #print("after:",state, action, reward, next_state, done)
            e = self.experience(state, action, reward, next_state, done, goal)
            self.memory.append(e)

    
    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx*self.n_step_buffer[idx][2]

        #return first of the experience state, action, expected return from the action, final state, and done at the end
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4] 

    #adjusts the replay buffer for hindsight learning for the last inputted episode
    def new_goals(self, buffer_idx, buffer_size, local_buffer_transitions, local_buffer_qvalues, n):
        new_goals = []
        indexes = []
        #create new goals
        for idx, exp in enumerate(local_buffer_transitions[buffer_idx,:buffer_size]):
            for _ in range(n):
                random_index = random.randint(idx, buffer_size)
                indexes.append(random_index)
                transition = local_buffer_transitions[buffer_idx, random_index]
                new_goal = transition[0]
                new_goals.append(new_goal)
        for step in range(buffer_size):
            transition = local_buffer_transitions[buffer_idx, step]
            s = transition[0]
            a = np.argmax(local_buffer_qvalues[buffer_idx, step])
            ns = transition[3]
            for new_goal in new_goals:
                
                r = self.reward_function(transition[0], new_goal)
                
                if (next_state == new_goal).all():
                    d = 1
                else:
                    d = 0
                self.add(s, a, r, ns, d, new_goal)
        

    def reward_function(self, state, goal):
        return 0 if (state == goal).all() else -1


    def sample(self): #check that the sample stacking end up correct
        """Randomly sample a batch of experiences from memory. """
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        goals = torch.from_numpy(np.stack([e.goal for e in experiences if e is not None])).float().to(self.device)

        #print(states[1])
        #print(actions[1])
        #print(rewards[1])
        #print(next_states[1])
        #print(dones[1])
        #print(goals[1])
        return (states, actions, rewards, next_states, dones, goals)


    def __len__(self):
        """Return current size of memory"""
        return len(self.memory)