"""
Module to define classes for replay memory
"""
import random
from distributed.actor import Transition


class ReplayMemory:
    """
    Simple uniform replay memory class.
    The data is stored in a simple linear container and is sampled
    with uniform probability from it.
    """
    def __init__(self, memory_size):
        self.memory = [Transition(None, None, None, None, None)] * memory_size
        self.memory_size = memory_size
        self.current_num_objects = 0
        self.is_full_memory = False

    def save(self, obj):
        """
        Save data objects. 
        """
        self.memory[self.current_num_objects] = obj
        self.current_num_objects = (self.current_num_objects + 1) % self.memory_size

        if self.current_num_objects >= self.memory_size - 1:
            self.is_full_memory = True

    def sample(self, batch_size):
        """
        Sample a batch of data.
        """
        valid_max_index = (
            self.memory_size if self.is_full_memory else self.current_num_objects
        )
        if valid_max_index < batch_size:
            return None, None, None, None
        transitions_and_priorities = random.sample(
            self.memory[:valid_max_index], batch_size
        )
        transitions_and_priorities = list(zip(*transitions_and_priorities))

        transitions = transitions_and_priorities[0]
        priorities = transitions_and_priorities[1]
        # return priorities for conformity reasons
        return transitions, None, None, priorities

    def __len__(self):
        return self.memory_size
