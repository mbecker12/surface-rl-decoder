import random
from distributed.actor import Transition


class ReplayMemory:
    def __init__(self, memory_size):
        # TODO: allocate sufficient memory for transition objects or tuples
        self.memory = [Transition(None, None, None, None, None)] * memory_size
        self.memory_size = memory_size
        self.index = 0
        self.is_full_memory = False

    def save(self, obj):
        self.memory[self.index] = obj
        self.index = (self.index + 1) % self.memory_size

        if self.index >= self.memory_size - 1:
            self.is_full_memory = True

    def sample(self, batch_size):
        valid_max_index = self.memory_size if self.is_full_memory else self.index
        print(f"{valid_max_index=}, {batch_size=}, {self.memory_size=}")
        if valid_max_index < batch_size:
            return None, None, None
        return random.sample(self.memory[:valid_max_index], batch_size), None, None

    def __len__(self):
        return self.memory_size
