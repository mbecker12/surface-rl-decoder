# TODO fill with tests
from distributed.sum_tree import SumTree
import numpy as np

def test_sum_tree_init():
    max_size = 100
    st = SumTree(max_size)

    iterations = 100
    random_data = np.random.random_sample(iterations)
    random_values = (np.random.random_sample(iterations) * 99) + 1
    for i in range(iterations):
        obj = {"data": random_data[i]}
        value = int(random_values[i])

        st.add(obj, value)

    assert st.filled_size() == iterations


def test_get_val():
    max_size = 200
    st = SumTree(max_size)

    iterations = 500
    random_data = np.random.random_sample(iterations)
    random_values = (np.random.random_sample(iterations) * 99) + 1
    for i in range(iterations):
        obj = {"data": random_data[i]}
        value = int(random_values[i])

        st.add(obj, value)

    for idx in range(0, max_size, 3):
        st.get_val(idx)

    for idx in range(-60, 0, 4):
        st.get_val(idx)

    find_return = st.find(0.1)

    value_at_find = st.get_val(find_return[-1])
    assert value_at_find == find_return[1]


if __name__ == "__main__":
    test_sum_tree_init()
    test_get_val()
