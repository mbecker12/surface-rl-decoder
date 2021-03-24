"""
Implementation of prioritized experience replay as an alternative to
the linear, standard replay memory.
This implementation makes it more likely for
high-loss events to be sampled so that the model
can exncounter more events which it can learn a lot from.
"""
from time import time
import random
from collections import namedtuple
import numpy as np
from distributed.sum_tree import SumTree

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "terminal"]
)

EPSILON = 1e-16


class PrioritizedReplayMemory:
    """The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """

    def __init__(self, memory_size, alpha):
        """Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        alpha: float
            exponent determine how much prioritization.
            Prob_i sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha

    def save(self, data, priority):
        """Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority ** self.alpha)

    def sample(self, batch_size, beta, tensorboard=None, verbosity=None):
        """The method return samples randomly.

        Parameters
        ----------
        batch_size: batch_size to be sampled
        beta: float, PER parameter
        tensorboard: (optional)(torch.utils.SummaryWriter)
            tensorboard instance for logging/monitoring
        verbosity: (optional)(int) verbosity level

        Returns
        -------
        out:
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        priorities:
            list of priorities
        """

        if self.tree.filled_size() < batch_size:
            return None, None, None, None

        out = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        i = 0
        while i < batch_size:
            rand = random.random()
            try:
                data, priority, index = self.tree.find(rand)
                priorities[i] = priority

                _weight = (
                    (1.0 / self.memory_size / priority) ** beta
                    if priority > 1e-16
                    else 0.0
                )
                weights[i] = _weight
                indices[i] = index
                out.append(data)
                self.priority_update([index], [0])  # To avoid duplicating
            except AssertionError as _:
                print(
                    "Caught AssertionError while trying to sample from replay memory. "
                    "Skipping to new sample."
                )
                continue
            else:
                i += 1

        if tensorboard is not None:
            if verbosity >= 4:
                current_time = time()
                tensorboard.add_histogram(
                    "per/sampled_priorities",
                    np.array(priorities, dtype=np.float32),
                    walltime=int(current_time * 1000),
                )
                tensorboard.add_histogram(
                    "per/sampled_weights",
                    np.array(weights, dtype=np.float32),
                    walltime=int(current_time * 1000),
                )
                tensorboard.add_histogram(
                    "per/sampled_indices",
                    np.array(indices, dtype=np.float32),
                    walltime=int(current_time * 1000),
                )

        self.priority_update(indices, priorities)  # Revert priorities

        weights_max = np.max(weights)

        if weights_max == 0:
            weights = np.zeros(batch_size, dtype=np.float64)
        else:
            weights_max_inv = np.float64(1.0 / weights_max)
            weights = weights * weights_max_inv

        return out, weights, indices, priorities

    def priority_update(self, indices, priorities):
        """Update the samples' priority.

        Parameters
        ----------
        indices:
            list of sample indices
        """
        for i, prio in zip(indices, priorities):
            self.tree.val_update(i, prio ** self.alpha)

    def reset_alpha(self, alpha):
        """Reset an exponent alpha.

        Parameters
        ----------
        alpha: float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [
            self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())
        ]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def print_tree(self):
        """
        Print a simple representation of the sum tree.
        """
        self.tree.print_tree()

    def filled_size(self):
        """
        Return the number of elements stored in the sum tree.
        """
        return self.tree.filled_size()
