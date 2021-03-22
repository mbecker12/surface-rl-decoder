"""
Implementation of prioritized experience replay as an alternative to
the linear, standard replay memory.
This implementation makes it more likely for
high-loss events to be sampled so that the model
can exncounter more events which it can learn a lot from.
"""
import traceback
import random
from collections import namedtuple
from distributed.sum_tree import SumTree

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "terminal"]
)

EPSILON = 1e-16


class PrioritizedReplayMemory(object):
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
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
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

    def sample(self, batch_size, beta):
        """The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < batch_size:
            return None, None, None, None

        out = []
        indices = []
        weights = []
        priorities = []

        i = 0
        while i < batch_size:
            rand = random.random()
            try:
                data, priority, index = self.tree.find(rand)
                priorities.append(priority)

                _weight = (
                    (1.0 / self.memory_size / priority) ** beta
                    if priority > 1e-16
                    else 0.0
                )
                weights.append(_weight)
                indices.append(index)
                out.append(data)
                self.priority_update([index], [0])  # To avoid duplicating
            except AssertionError as _:
                print(
                    "Caught AssertionError while trying to sample from replay memory. Skipping to new sample."
                )
                continue
            else:
                i += 1

        self.priority_update(indices, priorities)  # Revert priorities

        weights_max = max(weights)
        weights_max_inv = float(1.0 / weights_max)

        if weights_max == 0:
            weights = [0.0 for w in weights]
        else:
            weights = [
                float(i * weights_max_inv) for i in weights
            ]  # Normalize for stability

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
