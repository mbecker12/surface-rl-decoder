from src.surface_rl_decoder.surface_code import SurfaceCode
from src.surface_rl_decoder.surface_code_util import MAX_ACTIONS
from tests.data_episode_test import _actions


def test_full_action_history(sc):
    sc.actual_errors[:, 0, 0] = 1

    actions = [(0, 1, 1) for i in range(MAX_ACTIONS)]
    assert sc.current_action_index == 0

    for i, action in enumerate(actions):
        j = i + 1
        _, reward, terminal, _ = sc.step(action)
        assert sc.current_action_index == j, sc.current_action_index
        if j < MAX_ACTIONS:
            assert not terminal, j
            assert reward == 0
        else:
            assert terminal
            # negative reward for not being in ground state
            assert reward < 0

    assert not sc.ground_state
    assert sc.current_action_index == MAX_ACTIONS


def test_full_action_history_ground_state(
    configure_env, restore_env, seed_surface_code
):
    original_depth, original_size, original_error_channel = configure_env()

    sc = SurfaceCode()
    seed_surface_code(sc, 42, 0.1, 0.1, "dp")

    for action in _actions:
        sc.step(action)

    terminal = False
    while not terminal:
        _, reward, terminal, _ = sc.step(action=(3, 4, 0))

    assert sc.ground_state, sc.state
    assert reward > 0, sc.qubits
    restore_env(original_depth, original_size, original_error_channel)
