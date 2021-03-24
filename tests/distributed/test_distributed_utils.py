import numpy as np
import pytest
import torch
from src.distributed.util import (
    action_to_q_value_index,
    anneal_factor,
    q_value_index_to_action,
    select_actions,
)


def test_actor_utils(init_testing_model, load_model_config):
    system_size = 5
    stack_depth = 8
    state_size = system_size + 1
    epsilon = 0.25
    model_name = "dummy_agent"
    model_config = load_model_config("dummy_agent.json", model_name)

    model = init_testing_model(
        {"model_name": model_name, "model_config": model_config},
        state_size=state_size,
        stack_depth=stack_depth,
    )

    for batch_size in (1, 2, 16, 32, 100, 512):
        print(f"{batch_size=}")
        state = np.random.randint(
            0, 2, (batch_size, stack_depth, state_size, state_size), dtype=np.uint8
        )
        _state = torch.tensor(state, dtype=torch.float32)
        actions, q_values = select_actions(
            _state, model, system_size, num_actions_per_qubit=3, epsilon=epsilon
        )
        assert len(actions) == batch_size, actions
        assert q_values.shape == (batch_size, 3 * system_size * system_size + 1)


def test_action_to_idx():
    system_size = 7
    for _ in range(200):
        x_coord = np.random.randint(0, system_size)
        y_coord = np.random.randint(0, system_size)
        operator = np.random.randint(1, 5)
        action = (x_coord, y_coord, operator)
        index = action_to_q_value_index(action, system_size)
        assert 0 <= index <= 3 * system_size * system_size + 1


def test_action_and_idx():
    system_size = 5
    for y in range(system_size):
        for x in range(system_size):
            for ac in (1, 2, 3, 4):
                idx = action_to_q_value_index((x, y, ac), system_size)

                action = q_value_index_to_action(idx, system_size)
                if ac in (1, 2, 3):
                    assert action == (x, y, ac)
                else:
                    assert action[-1] == ac

    with pytest.raises(Exception):
        action_to_q_value_index((-9, -6, 0), system_size)

    with pytest.raises(AssertionError):
        action_to_q_value_index((-9, -6, 1), system_size)

    with pytest.raises(Exception):
        action_to_q_value_index((-9, -6, 895678), system_size)

    with pytest.raises(Exception):
        q_value_index_to_action(-2, system_size)

    with pytest.raises(Exception):
        q_value_index_to_action(999999999999999, system_size)

    for i in range(3 * system_size * system_size + 1):
        action = q_value_index_to_action(i, system_size)
        x, y, ac = action

        idx = action_to_q_value_index(action, system_size)
        assert idx == i


def test_annealing():
    timesteps = 1250

    min_val = 0.0
    for i in range(timesteps):
        factor = anneal_factor(timesteps=i, decay_factor=0.99, min_value=min_val)
        if i < 5:
            assert pytest.approx(factor, abs=0.05) == 1.0
    assert pytest.approx(factor, abs=1e-5) == min_val

    min_val = 0.3
    for i in range(timesteps):
        factor = anneal_factor(timesteps=i, decay_factor=0.99, min_value=min_val)
        if i < 5:
            assert pytest.approx(factor, abs=0.05) == 1.0
    assert pytest.approx(factor, abs=1e-5) == min_val

    max_val = 1.0
    base_factor = 0.4
    for i in range(timesteps):
        factor = anneal_factor(
            timesteps=i,
            decay_factor=1.01,
            min_value=min_val,
            max_value=max_val,
            base_factor=base_factor,
        )

        if i < 5:
            assert pytest.approx(factor, abs=0.05) == base_factor
    assert pytest.approx(factor, abs=1e-6) == max_val

    max_val = 1.0
    base_factor = 0.4
    timesteps = 50000
    for i in range(timesteps):
        factor = anneal_factor(
            time_difference=i,
            decay_factor=1.00002,
            min_value=min_val,
            max_value=max_val,
            base_factor=base_factor,
        )

        if i < 5:
            assert pytest.approx(factor, abs=0.05) == base_factor
    assert factor > base_factor
    assert pytest.approx(factor, abs=1e-6) == max_val
