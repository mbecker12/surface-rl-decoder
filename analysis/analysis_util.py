from copy import deepcopy
import os
import numpy as np
import torch
from agents.base_agent import BaseAgent
from distributed.environment_set import EnvironmentSet
from distributed.util import q_value_index_to_action, select_actions
from surface_rl_decoder.surface_code import SurfaceCode
from surface_rl_decoder.surface_code_util import TERMINAL_ACTION, check_final_state

def analyze_succesful_episodes(
    model: BaseAgent,
    environment_def,
    total_n_episodes=128,
    epsilon=0.0,
    max_num_of_steps=32,
    discount_factor_gamma=0.95,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.3,
    punish_repeating_actions=0,
    p_err=0.0,
    p_msmt=0.0,
    verbosity=0,
    rl_type="q",
    code_size=None,
    stack_depth=None,
    device="cpu"
):

    model.eval()

    if code_size is not None:
        os.environ["CONFIG_ENV_SIZE"] = str(code_size)
    if stack_depth is not None:
        os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)

    if environment_def is None or environment_def == "":
        environment_def = SurfaceCode(code_size=code_size, stack_depth=stack_depth)
    
    env_set = EnvironmentSet(environment_def, total_n_episodes)
    code_size = env_set.code_size
    stack_depth = env_set.stack_depth
    states = env_set.reset_all(
            np.repeat(p_err, total_n_episodes), np.repeat(p_msmt, total_n_episodes)
        )

    global_episode_steps = 0

    counter_logical_errors = 0
    counter_syndrome_left = 0
    counter_successful_episodes = 0
    counter_ground_state = 0
    counter_solved_w_syndrome_left = 0

    ignore_episodes = set()
    n_steps = np.zeros(total_n_episodes)
    while global_episode_steps <= max_num_of_steps:
        global_episode_steps += 1
        for i in range(total_n_episodes):
            if i not in ignore_episodes:
                n_steps[i] += 1

        torch_states = torch.tensor(states, dtype=torch.float32).to(device)
        if "q" in rl_type.lower():
            actions, tmp_q_values = select_actions(
                torch_states, model, code_size, epsilon=epsilon
            )
        elif "ppo" in rl_type.lower():
            if epsilon == 1:
                actions = model.select_greedy_action_ppo(
                    torch_states, return_logits=False, return_values=False
                )
            else:
                actions = model.select_action_ppo(
                    torch_states, return_logits=False, return_values=False
                )

            actions = np.array(
                [q_value_index_to_action(action, code_size) for action in actions]
            )
            
        next_states, rewards, terminals, _ = env_set.step(
            actions,
            discount_intermediate_reward=discount_intermediate_reward,
            annealing_intermediate_reward=annealing_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
        )
        
        non_terminal_episodes = np.where(actions[:, -1] != TERMINAL_ACTION)[0]

        if np.any(terminals):
            indices = np.argwhere(terminals).flatten()
            for i in indices:
                if i in ignore_episodes:
                    continue
                ignore_episodes.add(i)

                _, _ground_state, (n_syndromes, n_loops) = check_final_state(
                    env_set.environments[i].actual_errors,
                    env_set.environments[i].actions,
                    env_set.environments[i].vertex_mask,
                    env_set.environments[i].plaquette_mask,
                )

                if n_syndromes > 0:
                    counter_syndrome_left += 1
                if n_loops > 0:
                    counter_logical_errors += 1
                # assume that a few syndromes left don't cause much harm
                # and are in fact okay for our purposes
                if _ground_state and n_loops == 0:
                    counter_successful_episodes += 1

                    if n_syndromes > 0:
                        counter_solved_w_syndrome_left += 1

                if _ground_state:
                    counter_ground_state += 1

        if np.all(terminals):
            break

        states = next_states
        env_set.states = deepcopy(states)

    for i in range(total_n_episodes):
        if i in ignore_episodes:
            continue
        
        _, _ground_state, (n_syndromes, n_loops) = check_final_state(
            env_set.environments[i].actual_errors,
            env_set.environments[i].actions,
            env_set.environments[i].vertex_mask,
            env_set.environments[i].plaquette_mask,
        )

        if n_syndromes > 0:
            counter_syndrome_left += 1
        if n_loops > 0:
            counter_logical_errors += 1
        # assume that a few syndromes left don't cause much harm
        # and are in fact okay for our purposes
        if _ground_state and n_loops == 0:
            counter_successful_episodes += 1

            if n_syndromes > 0:
                counter_solved_w_syndrome_left += 1

        if _ground_state:
            counter_ground_state += 1

    return (
        total_n_episodes,
        counter_successful_episodes,
        counter_logical_errors,
        counter_syndrome_left,
        counter_ground_state,
        counter_solved_w_syndrome_left,
        n_steps
    )
