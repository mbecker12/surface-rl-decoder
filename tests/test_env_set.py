# from os import environ
from copy import deepcopy
import numpy as np
from src.distributed.environment_set import EnvironmentSet
from src.surface_rl_decoder.surface_code_util import STATE_MULTIPLIER, TERMINAL_ACTION


def test_init(sc):
    n_env = 10
    environment_set = EnvironmentSet(sc, n_env)
    assert environment_set.num_environments == n_env
    assert environment_set.stack_depth == sc.stack_depth
    # pylint: disable=protected-access
    assert environment_set.states.shape == (
        n_env,
        sc.stack_depth,
        sc.syndrome_size,
        sc.syndrome_size,
    )


def test_reset_all(env_set):
    n_envs = env_set.num_environments
    states = deepcopy(env_set.states)
    new_states = env_set.reset_all()

    assert not np.all(states == new_states), new_states

    for i in range(n_envs):
        assert np.max(env_set.environments[i].state) == np.max(new_states[i])
        assert np.all(env_set.environments[i].state == new_states[i])
        assert id(env_set.environments[i].state) != id(new_states[i])


def test_reset_all_w_new_probabilities(env_set):
    states = deepcopy(env_set.states)
    p_error = [0.5] * env_set.num_environments
    p_msmt = [0.5] * env_set.num_environments
    new_states = env_set.reset_all(p_error=p_error, p_msmt=p_msmt)

    assert not np.all(states == new_states), new_states


def test_step(env_set):
    old_states = env_set.reset_all()
    n_envs = env_set.num_environments

    actions = np.random.randint(0, 4, size=(n_envs, 3))

    new_states, *_ = env_set.step(actions)

    assert not np.all(old_states == new_states), new_states

    for i in range(n_envs):
        assert np.max(env_set.environments[i].state) == np.max(new_states[i])
        assert np.all(env_set.environments[i].state == new_states[i])
        assert id(env_set.environments[i].state) != id(new_states[i])


def test_terminal(env_set):
    old_states = env_set.reset_all()
    n_envs = env_set.num_environments

    actions = np.ones((n_envs, 3)) * TERMINAL_ACTION
    new_states, _, terminals, _ = env_set.step(actions)

    assert np.max(old_states) == np.max(new_states)
    assert np.all(terminals)
    assert np.all(old_states == new_states)


def test_reset_terminal(env_set):
    env_set.reset_all()
    n_envs = env_set.num_environments

    actions = np.zeros((n_envs, 3), dtype=np.uint8)

    terminal_idx = [1, 4]
    for i in terminal_idx:
        actions[i] = (99, 99, TERMINAL_ACTION)

    new_states, _, terminals, _ = env_set.step(actions)

    _terminals_idx = np.where(terminals)[0]
    assert np.all(_terminals_idx == terminal_idx)

    for i in range(n_envs):
        assert np.all(env_set.environments[i].state == new_states[i])
        assert id(env_set.environments[i].state) != id(new_states[i])

    p_error = [0.5] * env_set.num_environments
    p_msmt = [0.5] * env_set.num_environments

    update_states = env_set.reset_terminal_environments(
        terminal_idx, p_error=p_error, p_msmt=p_msmt
    )

    assert not np.all(update_states[terminal_idx] == new_states[terminal_idx])

    terminal_idx_set = set(terminal_idx)
    all_indices = set(range(n_envs))
    diff_set = all_indices.difference(terminal_idx_set)

    for i in diff_set:
        assert i in (0, 2, 3)
    for i in terminal_idx:
        assert i not in diff_set

    assert np.all(update_states[list(diff_set)] == new_states[list(diff_set)])


def test_reset_terminal_change_probabilities(env_set):
    env_set.reset_all()
    n_envs = env_set.num_environments

    actions = np.zeros((n_envs, 3), dtype=np.uint8)

    terminal_idx = [1, 4]
    for i in terminal_idx:
        actions[i] = (99, 99, TERMINAL_ACTION)

    new_states, _, terminals, _ = env_set.step(actions)

    _terminals_idx = np.where(terminals)[0]
    assert np.all(_terminals_idx == terminal_idx)

    update_states = env_set.reset_terminal_environments(terminal_idx)

    assert not np.all(update_states[terminal_idx] == new_states[terminal_idx])

    terminal_idx_set = set(terminal_idx)
    all_indices = set(range(n_envs))
    diff_set = all_indices.difference(terminal_idx_set)

    for i in diff_set:
        assert i in (0, 2, 3)
    for i in terminal_idx:
        assert i not in diff_set

    assert np.all(update_states[list(diff_set)] == new_states[list(diff_set)])


def test_indepence_of_envs(env_set):
    env_set.reset_all()
    n_envs = env_set.num_environments

    for _ in range(5):
        actions = np.random.randint(0, 4, size=(n_envs, 3))
        env_set.step(actions)

    for i in range(n_envs):
        for j in range(i + 1, n_envs):
            assert not np.all(env_set.states[i] == env_set.states[j])

    n_terminal_states = 2
    terminal_actions = np.ones((n_terminal_states, 3), dtype=np.uint8) * TERMINAL_ACTION
    non_terminal_actions = np.ones((n_envs - n_terminal_states, 3), dtype=np.uint8)
    all_actions = np.concatenate((terminal_actions, non_terminal_actions))

    assert all_actions.shape == (n_envs, 3)

    _, _, terminals, _ = env_set.step(all_actions)
    assert np.all(terminals[:n_terminal_states])
    assert not np.any(terminals[n_terminal_states:])
