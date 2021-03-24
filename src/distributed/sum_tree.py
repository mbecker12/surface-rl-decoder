"""
Implementation of a sum tree
to be used in prioritized replay memory.

Based on https://github.com/Lindeby/toric-RL-decoder/blob/master/src/SumTree.py
"""
import math


class SumTree:
    """
    Initialize a sum tree object to store
    data points in a tree structure.
    Used for prioritized replay memory
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        """
        Add an item to the tree.

        Parameters
        ==========
        contents: The data object to store at a location in the tree
        value: Value corresponding to the contents to add to the tree,
            used as priorization value
        """
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        """
        Return the value at a spcific index

        Parameters
        ==========
        index: requested index

        Returns
        =======
        value: Value at requested index
        """
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        """
        Update the value at a specific index

        Parameters
        ==========
        index: The tree index for the new value
        value: New value
        """
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        """
        # No idea, please refer to proper documentation on sum trees
        """
        # TODO docu
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        """
        Find an element in the tree based on an input value.
        If norm is True, the input value is expected to be between 0 and 1.
        """
        assert value >= 0.0, f"{value=}"
        if norm:
            value *= self.tree[0]
        return self._find(value, 0, original_value=value)

    def _find(self, value, index, original_value=0):
        """
        No idea, please refer to proper documentation on sum trees
        """
        # TODO docu
        if 2 ** (self.tree_level - 1) - 1 <= index:
            data_index = index - (2 ** (self.tree_level - 1) - 1)
            assert 0 <= data_index < len(self.data), (
                f"{original_value=}, {self.tree[0]=}, "
                f"{(original_value / self.tree[0])=}, {value=}, "
                f"{data_index=}, {len(self.data)=}, {self.tree_level=}, {index=}\n"
            )

            return (
                self.data[data_index],
                self.tree[index],
                data_index,
            )

        left = self.tree[2 * index + 1]

        # pylint: disable=no-else-return
        if value <= left:
            return self._find(value, 2 * index + 1, original_value=original_value)
        else:
            return self._find(
                value - left, 2 * (index + 1), original_value=original_value
            )

    def print_tree(self):
        """
        Print the current realization of the tree.
        """
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=" ")
            print()

    def filled_size(self):
        """
        Return the number of elements currently in the tree.
        """
        return self.size
