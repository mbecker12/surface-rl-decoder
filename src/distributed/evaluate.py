"""
Define an evaluation routine to keep track of the agent's ability
to decode syndromes
"""
import logging
from typing import Dict, Tuple
from distributed.eval_util import (
    RESULT_KEY_EPISODE,
    RESULT_KEY_STEP,
    RESULT_KEY_P_ERR,
    run_evaluation_in_batches,
)
from distributed.learner_util import safe_append_in_dict
from evaluation.batch_evaluation import (
    RESULT_KEY_COUNTS,
    RESULT_KEY_ENERGY,
    RESULT_KEY_INCREASING,
    RESULT_KEY_Q_VALUE_STATS,
    RESULT_KEY_RATES,
    batch_evaluation,
)
from surface_rl_decoder.surface_code import SurfaceCode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

# pylint: disable=too-many-arguments, too-many-locals
def evaluate(
    model,
    environment_def,
    device,
    p_error_list,
    p_msmt_list,
    num_of_random_episodes=120,
    num_of_user_episodes=8,
    epsilon=0.0,
    max_num_of_steps=50,
    discount_factor_gamma=0.9,
    annealing_intermediate_reward=1.0,
    discount_intermediate_reward=0.3,
    punish_repeating_actions=0,
    plot_one_episode=True,
    verbosity=0,
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate the current policy.
    Loop through different levels of error rates for both
    physical and syndrome measurement errors.

    Parameters
    ==========
    model: (subclass of torch.nn.Module) Neural network model
    environment_def: (str or gym.Env) either environment name, or object
    device: torch.device
    num_of_random_episodes: number of episodes to with fully randomly generated states
    num_of_user_episodes: number of user-creates and predefined episodes,
        taken from create_user_eval_state(), hence this number is limited by the number of
        availabe examples in the helper function
    epsilon: probability of the agent choosing a random action
    max_num_of_steps: maximum number of steps per environment
    plot_one_episode: whether or not to render an example episode
    discount_factor_gamma: gamma / discount factor in reinforcement learning
    p_error_list: list of error rates for physical errors
    p_msmt_list: list of error rates for syndrome measurement errors
    annealing_intermediate_reward: (optional) variable that should decrease over time during
        a training run to decrease the effect of the intermediate reward
    punish_repeating_actions: (optional) (1 or 0) flag acting as multiplier to
        enable punishment for repeating actions that already exist in the action history
    discount_intermediate_reward: (optional) discount factor determining how much
        early layers should be discounted when calculating the intermediate reward
    verbosity: (int) verbosity level

    Returns
    =======
    (Dict, Dict, Dict): dictionaries with evaluation metrics for three different categories
    """

    model.eval()

    if environment_def is None or environment_def == "":
        environment_def = SurfaceCode()
    code_size = environment_def.code_size
    assert (
        code_size % 2 == 1
    ), "System size (i.e. number of qubits) needs to be an odd number."

    final_result_dict = {
        RESULT_KEY_EPISODE: {},
        RESULT_KEY_Q_VALUE_STATS: {},
        RESULT_KEY_ENERGY: {},
        RESULT_KEY_COUNTS: {},
        RESULT_KEY_RATES: {},
    }

    for i_err_list, p_error in enumerate(p_error_list):
        eval_results, all_q_values = batch_evaluation(
            model,
            environment_def,
            device,
            num_of_random_episodes=num_of_random_episodes,
            num_of_user_episodes=num_of_user_episodes,
            epsilon=epsilon,
            max_num_of_steps=max_num_of_steps,
            discount_factor_gamma=discount_factor_gamma,
            annealing_intermediate_reward=annealing_intermediate_reward,
            discount_intermediate_reward=discount_intermediate_reward,
            punish_repeating_actions=punish_repeating_actions,
            p_err=p_error,
            p_msmt=p_msmt_list[i_err_list],
            verbosity=verbosity,
        )

        for category_name, category in eval_results.items():
            for key, val in category.items():
                final_result_dict[category_name] = safe_append_in_dict(
                    final_result_dict[category_name], key, val
                )

    # end for; error_list

    return (final_result_dict, all_q_values)
