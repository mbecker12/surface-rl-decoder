# from os import environ
import numpy as np
from copy import deepcopy
from src.distributed.environment_set import EnvironmentSet
from src.surface_rl_decoder.surface_code_util import TERMINAL_ACTION


def test_init(sc):
    n_env = 10
    environment_set = EnvironmentSet(sc, n_env)
    assert environment_set.num_environments == n_env
    assert environment_set.stack_depth == sc.stack_depth
    assert environment_set.states.shape == (
        n_env,
        sc.stack_depth,
        sc.syndrome_size,
        sc.syndrome_size,
    )


def test_reset_all(env_set):
    states = deepcopy(env_set.states)
    new_states = env_set.reset_all()

    assert not np.all(states == new_states), new_states


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


def test_terminal(env_set):
    old_states = env_set.reset_all()
    n_envs = env_set.num_environments

    actions = np.ones((n_envs, 3)) * TERMINAL_ACTION
    new_states, _, terminals, _ = env_set.step(actions)

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
