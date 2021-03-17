import numpy as np
from distributed.util import compute_priorities
from distributed.prioritized_replay_memory import PrioritizedReplayMemory, Transition

stack_depth = 8
state_size = 6
transition_type = np.dtype(
    [
        ("state", (np.uint8, (stack_depth, state_size, state_size))),
        ("action", (np.uint8, 3)),
        ("reward", float),
        ("next_state", (np.uint8, (stack_depth, state_size, state_size))),
        ("terminal", bool),
    ]
)


def test_compute_priorities():
    num_environments = 17
    buffer_size = 10
    system_size = 5

    actions = np.random.randint(1, 5, size=(num_environments, buffer_size - 1, 3))
    rewards = np.random.randint(
        0, 2, size=(num_environments, buffer_size - 1), dtype=bool
    )
    qvalues = np.random.random_sample(
        size=(num_environments, buffer_size - 1, 3 * system_size * system_size + 1)
    )
    gamma = 1.0
    priorities = compute_priorities(actions, rewards, qvalues, gamma, system_size)
    assert priorities.shape == (
        num_environments,
        buffer_size - 1,
    )


def test_deterministic_priorities():
    num_environments = 4
    buffer_size = 3
    system_size = 3
    gamma = 1

    actions = np.array(
        [
            [[0, 0, 1], [0, 1, 2]],
            [[1, 0, 1], [1, 1, 2]],
            [[2, 0, 1], [2, 1, 2]],
            [[0, 0, 1], [0, 1, 2]],
        ]
    )

    rewards = np.array([[100, 50], [20, 10], [100, 50], [100, 50]])

    max_q = 3 * system_size * system_size
    qvalues = np.array(
        [
            [
                np.arange(max_q + 1),
                np.arange(max_q + 1),
            ],
            [
                np.arange(max_q + 1),
                np.arange(max_q + 1),
            ],
            [
                np.arange(max_q + 1),
                np.arange(max_q + 1),
            ],
            [
                np.arange(max_q + 1),
                np.arange(max_q + 1),
            ],
        ]
    )

    qvalues = np.multiply(qvalues, 2)

    assert actions.shape == (num_environments, buffer_size - 1, 3)
    assert rewards.shape == (num_environments, buffer_size - 1)
    assert qvalues.shape == (
        num_environments,
        buffer_size - 1,
        3 * system_size ** 2 + 1,
    )

    # priorities = TD error
    # R + discount*Qns_max - Qv

    expected_priorities = np.array(
        [
            [100 + 1 * 2 * (max_q - 0), 50 + 1 * 2 * (max_q - 10)],
            [20 + 1 * 2 * (max_q - 3), 10 + 1 * 2 * (max_q - 13)],
            [100 + 1 * 2 * (max_q - 6), 50 + 1 * 2 * (max_q - 16)],
            [100 + 1 * 2 * (max_q - 0), 50 + 1 * 2 * (max_q - 10)],
        ]
    )

    priorities = compute_priorities(actions, rewards, qvalues, gamma, system_size)
    assert np.all(
        priorities == expected_priorities
    ), f"{priorities=}, {expected_priorities=}"


def test_sampling():
    buffer_size = 100
    memory_size = 100
    alpha = 0.5
    num_environments = 25

    system_size = 5
    replay_memory = PrioritizedReplayMemory(memory_size, alpha)

    actions = np.random.randint(1, 5, size=(num_environments, buffer_size - 1, 3))
    rewards = np.random.randint(
        0, 2, size=(num_environments, buffer_size - 1), dtype=bool
    )
    qvalues = np.random.random_sample(
        size=(num_environments, buffer_size - 1, 3 * system_size * system_size + 1)
    )
    gamma = 1.0
    priorities = compute_priorities(actions, rewards, qvalues, gamma, system_size)

    states = np.random.randint(
        0,
        2,
        size=(num_environments, buffer_size, stack_depth, state_size, state_size),
        dtype=np.uint8,
    )
    next_states = np.random.randint(
        0,
        2,
        size=(num_environments, buffer_size, stack_depth, state_size, state_size),
        dtype=np.uint8,
    )
    terminals = np.random.randint(
        0, 2, size=(num_environments, buffer_size), dtype=bool
    )

    transitions = np.asarray(
        [
            Transition(
                states[i][buf],
                actions[i][buf],
                rewards[i][buf],
                next_states[i][buf],
                terminals[i][buf],
            )
            for buf in range(buffer_size - 1)
            for i in range(num_environments)
        ],
        dtype=transition_type,
    )

    flat_priorities = priorities.flatten()
    assert flat_priorities.shape[0] == (buffer_size - 1) * num_environments
    assert len(transitions) == (buffer_size - 1) * num_environments, len(transitions)
    for i, _ in enumerate(transitions):
        replay_memory.save(transitions[i], flat_priorities[i])

    # replay_memory.tree.print_tree()
    n_elements = replay_memory.filled_size()
    assert n_elements >= buffer_size - 1

    batch_size = 32
    memory_beta = 0.8
    transitions, memory_weights, indices, priorities = replay_memory.sample(
        batch_size, memory_beta
    )

    assert len(transitions) == batch_size, len(transitions)
    assert len(transitions[0]) == 5, len(transitions[0])
    assert len(memory_weights) == batch_size, len(memory_weights)
    assert len(priorities) == batch_size, len(priorities)


if __name__ == "__main__":
    test_deterministic_priorities()
