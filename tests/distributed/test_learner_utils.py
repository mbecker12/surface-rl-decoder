import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from src.distributed.learner_util import data_to_batch, perform_q_learning_step


def test_data_to_batch():
    device = torch.device("cpu")
    syndrome_size = 6
    stack_depth = 12

    for batch_size in (1, 10, 16, 100):
        transitions = []
        for _ in range(batch_size):
            state = np.random.randint(
                0, 2, (stack_depth, syndrome_size, syndrome_size), dtype=np.uint8
            )
            action = np.random.randint(1, 5, size=(3,), dtype=np.uint8)
            next_state = np.random.randint(
                0, 2, (stack_depth, syndrome_size, syndrome_size), dtype=np.uint8
            )
            terminal = np.random.randint(0, 2, dtype=bool)
            reward = np.random.random_sample()
            tran = [state, action, reward, next_state, terminal]
            transitions.append(tran)

        weights = np.random.random_sample(batch_size)
        indices = None
        data = (transitions, weights, indices)
        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_terminal,
            memory_weights,
            indices,
        ) = data_to_batch(data, device)

        assert batch_state.shape == (
            batch_size,
            stack_depth,
            syndrome_size,
            syndrome_size,
        )
        assert batch_next_state.shape == (
            batch_size,
            stack_depth,
            syndrome_size,
            syndrome_size,
        )
        assert batch_action.shape == (batch_size, 3)
        assert batch_reward.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)


def test_q_learning_step(init_testing_model, load_model_config):
    state_size = 10
    stack_depth = 32
    batch_size = 64
    model_name = "dummy_agent"
    model_config = load_model_config("dummy_agent.json", model_name)

    policy_model = init_testing_model(
        {"model_name": model_name, "model_config": model_config},
        state_size=state_size,
        stack_depth=stack_depth,
    )

    target_model = init_testing_model(
        {"model_name": model_name, "model_config": model_config},
        state_size=state_size,
        stack_depth=stack_depth,
    )

    assert id(policy_model) != id(target_model)
    device = torch.device("cpu")
    criterion = nn.MSELoss(reduction="none")
    learning_rate = 1e-3
    optimizer = Adam(policy_model.parameters(), lr=learning_rate)

    for weights in (True, False):
        transitions = []
        for _ in range(batch_size):
            state = np.random.randint(
                0, 2, (stack_depth, state_size, state_size), dtype=np.uint8
            )
            action = np.random.randint(1, 5, size=(3,), dtype=np.uint8)
            next_state = np.random.randint(
                0, 2, (stack_depth, state_size, state_size), dtype=np.uint8
            )
            terminal = np.random.randint(0, 2, dtype=bool)
            reward = np.random.random_sample()
            tran = [state, action, reward, next_state, terminal]
            transitions.append(tran)
        if weights:
            memory_weights = np.random.random_sample(batch_size)
            memory_weights = torch.tensor(memory_weights, dtype=torch.float32)
        indices = None
        data = (transitions, memory_weights, indices)

        perform_q_learning_step(
            policy_model,
            target_model,
            device,
            criterion,
            optimizer,
            data,
            state_size - 1,
            batch_size,
            discount_factor=0.9,
        )
