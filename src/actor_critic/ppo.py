"""
Implementation of the general PPO class.
This class spawns the required helper classes for
the replay buffer and different workers.
Furthermore, this class implements the communication with the
replay buffer to request episode samples and performs the actual learner step.
"""
import json
import os
from time import time
import logging
import traceback
import nvgpu

# pylint: disable=not-callable
import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam, SGD
from torch.utils.tensorboard.writer import SummaryWriter
from actor_critic.ppo_env import MultiprocessEnv
from actor_critic.replay_buffer import EpisodeBuffer
from distributed.model_util import (
    choose_model,
    extend_model_config,
    load_model,
    save_model,
)
from distributed.util import action_to_q_value_index
from distributed.learner_util import log_evaluation_data, transform_list_dict
from distributed.io_util import monitor_cpu_memory, monitor_gpu_memory
from evaluation.batch_evaluation import RESULT_KEY_HISTOGRAM_Q_VALUES
from evaluation.evaluate import evaluate

EPS = 1e-16

# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-locals, too-many-statements, too-many-branches
class PPO:
    """
    Main class to run the PPO learning method.
    This will take care of initializing the replay buffer and worker
    environment which in turn will spawn multiple workers.

    Configuration of this and all daughter processes may be done
    through a helper function,
    this class takes as input multiple dictionaries which said helper
    function should provide.
    """

    def __init__(
        self, worker_args, mem_args, learner_args, env_args, global_config, queues
    ) -> None:

        self.num_cuda_workers = env_args["num_cuda_actors"]
        self.num_cpu_workers = env_args["num_cpu_actors"]
        self.num_workers = self.num_cuda_workers + self.num_cpu_workers

        self.stack_depth = env_args["stack_depth"]
        self.code_size = env_args["code_size"]
        self.syndrome_size = self.code_size + 1

        self.device = learner_args["device"]
        self.learning_rate = learner_args.get("learning_rate")
        self.max_episodes = learner_args["max_episodes"]

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ppo")
        self.benchmark = learner_args["benchmarking"]
        self.verbosity = learner_args["verbosity"]
        if self.verbosity >= 4:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.io_verbosity = mem_args["verbosity"]

        summary_path = env_args["summary_path"]
        summary_date = env_args["summary_date"]
        load_model_flag = learner_args["load_model"]
        old_model_path = learner_args["old_model_path"]
        save_model_path = learner_args["save_model_path"]
        model_name = learner_args["model_name"]
        model_config = learner_args["model_config"]
        base_model_config_path = learner_args["base_model_config_path"]
        base_model_path = learner_args["base_model_path"]
        use_transfer_learning = learner_args["use_transfer_learning"]
        rl_type = learner_args["rl_type"]

        self.save_model_path_date = os.path.join(
            save_model_path,
            str(self.code_size),
            summary_date,
            f"{model_name}_{self.code_size}_{summary_date}.pt",
        )

        self.start_time = time()
        max_time_h = learner_args["max_time"]  # hours
        max_time_min = float(learner_args.get("max_time_minutes", 0))  # minutes
        self.max_time = max_time_h * 60 * 60  # seconds
        self.max_time += max_time_min * 60  # seconds

        self.heart = time()
        self.heartbeat_interval = 60  # seconds
        self.max_timesteps = learner_args["timesteps"]
        if self.max_timesteps == -1:
            self.max_timesteps = np.Infinity

        # initialize models and other learning gadgets
        model_config = extend_model_config(
            model_config, self.syndrome_size, self.stack_depth, device=self.device
        )

        # prepare Transfer learning, if enabled
        if use_transfer_learning:
            self.logger.info(f"Prepare transfer learning for d={self.code_size}.")
            with open(base_model_config_path, "r") as json_file:
                base_model_config = json.load(json_file)["simple_conv"]

            base_model_config = extend_model_config(
                base_model_config, self.syndrome_size, self.stack_depth, device=self.device
            )
        else:
            base_model_config = None

        model_config["rl_type"] = "ppo"
        self.combined_model = choose_model(
            model_name,
            model_config,
            model_path_base=base_model_path,
            model_config_base=base_model_config,
            transfer_learning=use_transfer_learning,
            rl_type=rl_type,
        )

        if load_model_flag:
            self.combined_model, self.optimizer, _ = load_model(
                self.combined_model,
                old_model_path,
                load_optimizer=True,
                load_criterion=True,
                optimizer_device=self.device,
                model_device=self.device,
                learning_rate=self.learning_rate,
            )
            self.logger.info(f"Loaded learner models from {old_model_path}")
        else:
            self.combined_model.to(self.device)

            self.optimizer = Adam(
                self.combined_model.parameters(), lr=self.learning_rate
            )

        # initialize tensorboard
        self.tensorboard = SummaryWriter(
            os.path.join(summary_path, str(self.code_size), summary_date, "learner")
        )
        tensorboard_string = "global config: " + str(global_config) + "\n"
        self.tensorboard.add_text("run_info/hyper_parameters", tensorboard_string)

        self.tensorboard_step = 0
        self.tensorboard_step_returns = 0
        self.received_data = 0

        self.policy_model_max_grad_norm = learner_args.get("policy_model_max_grad_norm")
        self.policy_clip_range = learner_args.get("policy_clip_range")
        self.policy_stopping_kl = learner_args.get("policy_stopping_kl")
        self.value_model_max_grad_norm = learner_args.get("value_model_max_grad_norm")
        self.value_clip_range = learner_args.get("value_clip_range")
        self.value_stopping_mse = learner_args.get("value_stopping_mse")
        self.entropy_loss_weight = torch.tensor(
            float(learner_args.get("entropy_loss_weight")),
            dtype=float,
            device=self.device,
        )
        self.value_loss_weight = torch.tensor(
            float(learner_args.get("value_loss_weight")),
            dtype=float,
            device=self.device,
        )
        self.discount_factor = learner_args["discount_factor"]
        self.optimization_epochs = learner_args["optimization_epochs"]
        self.batch_size = learner_args["batch_size"]

        self.eval_frequency = learner_args["eval_frequency"]
        self.p_error_list = learner_args["learner_eval_p_error"]
        self.p_msmt_list = learner_args["learner_eval_p_msmt"]
        self.learner_epsilon = learner_args["learner_epsilon"]

        assert self.num_workers > 1

        # extend buffer configuration
        mem_args["num_workers"] = self.num_workers
        mem_args["stack_depth"] = self.stack_depth
        mem_args["code_size"] = self.code_size
        mem_args["discount_factor"] = self.discount_factor
        mem_args["total_received_samples"] = 0
        mem_args["tensorboard"] = self.tensorboard

        try:
            nvgpu.gpu_info()

            self.gpu_available = True
        except FileNotFoundError as _:
            self.gpu_available = False

        self.nvidia_log_time = time()
        self.nvidia_log_frequency = mem_args["nvidia_log_frequency"]

        self.logger.info("Initialize Episode Buffer")
        self.episode_buffer = EpisodeBuffer(mem_args)

        self.logger.info("Set up Multi Process Environment")
        self.worker_queues = queues["worker_queues"]
        self.env_set = MultiprocessEnv(env_args, worker_args, queues)

        self.total_learner_recv_samples = 0

    def optimize_model(self, current_timestep):
        """
        Calculate the actual loss function value and perform the gradient
        descent step.
        To achieve that, samples are drawn from the replay buffer stacks
        and the required quantities, such as entropy, logpas, values, are
        calculated. This is done in multiple epochs, where each time a new
        sample is drawn from the replay buffer. This makes it possible
        to reuse worker episodes.

        Parameter
        =========
        current_timestep: current learner timestep; used for tensorboard logging
        """
        (
            states,
            actions,
            returns,
            gaes,
            logpas,
            values,
        ) = self.episode_buffer.get_stacks()

        self.combined_model.train()

        self.total_learner_recv_samples += states.shape[0]
        current_time = time()
        if self.io_verbosity:
            self.tensorboard.add_scalar(
                "io/learner_recv_samples",
                self.total_learner_recv_samples,
                current_timestep,
                current_time,
            )

            if self.verbosity >= 3:
                # np.choice(returns,
                random_sample_indices = np.random.choice(range(len(returns)), 10)
                for random_i in random_sample_indices:
                    self.tensorboard.add_scalars(
                        "episodes/returns",
                        {"return": returns[random_i]},
                        self.tensorboard_step_returns,
                        walltime=current_time,
                    )
                    self.tensorboard_step_returns += 1

        actions = torch.tensor(
            [action_to_q_value_index(action, self.code_size) for action in actions],
            device=self.combined_model.device,
        )

        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)
        n_samples = len(actions)

        if self.verbosity >= 8:
            self.logger.debug("start optimization loop")
        for _ in range(self.optimization_epochs):
            batch_idxs = np.random.choice(n_samples, self.batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            logpas_batch = logpas[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]

            (
                logpas_pred,
                entropies_pred,
                values_pred,
            ) = self.combined_model.get_predictions_ppo(states_batch, actions_batch)

            ratios = (logpas_pred - logpas_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(
                1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range
            )

            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight

            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(
                -self.value_clip_range, self.value_clip_range
            )
            v_loss = (returns_batch - values_pred).pow(2)
            v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
            value_loss = (
                torch.max(v_loss, v_loss_clipped).mul(0.5).mean()
                * self.value_loss_weight
            )

            self.optimizer.zero_grad()

            total_loss = value_loss + policy_loss + entropy_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.combined_model.parameters(), self.policy_model_max_grad_norm
            )
            self.optimizer.step()

            if self.verbosity >= 6:
                grad_string = ""
                for i, param in enumerate(self.combined_model.parameters()):
                    grad_string += f"{param.grad.data.sum().item():.2e}, "
                print(grad_string)
                if self.verbosity >= 7:
                    policy_params = list(self.combined_model.parameters())
                    n_layers = len(policy_params)
                    named_policy_params = self.combined_model.named_parameters()
                    for i, (par_name, param) in enumerate(named_policy_params):
                        if "transfomer.layers.0" in par_name:  # sic!
                            if "weight" in par_name:
                                try:
                                    print(f"{par_name}, {param[0][0]}, {param[-1][-1]}")
                                except:
                                    continue
                        if i in (
                            0,
                            int(n_layers // 2),
                            n_layers - 4,
                            n_layers - 3,
                            n_layers - 2,
                            n_layers - 1,
                        ):
                            if "weight" in par_name:
                                print(
                                    f"{par_name}, {param[0][0]}, {param.grad[0][0]}, {param[-1][-1]}, {param.grad[-1][-1]}"
                                )

        if self.verbosity >= 9:
            self.logger.debug("end optimization loop")

    def train(self, seed):
        """
        Set up the learning/training framework:
        Trigger the replay buffer to be filled by telling the
        workers to generate episodes.
        Those samples are then used to optimize the model,
        i.e. perform the actual gradient descent steps.
        """
        self.logger.debug("Inside training function")

        self.seed = seed
        self.gamma = self.discount_factor

        if self.seed != 0:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.episode_exploration = [], []
        self.evaluation_scores = []

        result = np.empty((self.max_episodes, 5))
        result[:] = np.nan

        performance_start = time()
        count_to_eval = 0
        eval_step = 0

        for t in range(self.max_timesteps):
            current_time = time()
            count_to_eval += 1

            if time() - self.start_time > self.max_time:
                self.logger.warning("Learner: time exceeded, aborting...")
                break

            (
                episode_timestep,
                episode_reward,
                episode_exploration,
                episode_seconds,
            ) = self.episode_buffer.fill(self.env_set, self.combined_model)

            if self.io_verbosity:
                if (
                    self.gpu_available
                    and self.nvidia_log_time > self.nvidia_log_frequency
                ):
                    monitor_gpu_memory(
                        self.tensorboard, current_time, performance_start, current_time
                    )
                if self.io_verbosity >= 3:
                    monitor_cpu_memory(
                        self.tensorboard, current_time, performance_start, current_time
                    )

            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)

            try:
                self.optimize_model(t)
            except TypeError as _:
                error_traceback = traceback.format_exc()
                self.logger.error("Caught exception in learning step")
                self.logger.error(error_traceback)
            except RuntimeError as rt_err:
                error_traceback = traceback.format_exc()
                self.logger.error(
                    "Caught runtime error in learning step. Terminating program..."
                )
                self.logger.error(error_traceback)
                break

            self.episode_buffer.clear()

            # stats
            if self.eval_frequency != -1 and count_to_eval >= self.eval_frequency:
                self.logger.info(f"Start Evaluation, Step {t+1}")
                count_to_eval = 0

                if self.verbosity >= 7:
                    policy_params = list(self.combined_model.parameters())
                    n_layers = len(policy_params)
                    named_policy_params = self.combined_model.named_parameters()
                    for i, (par_name, param) in enumerate(named_policy_params):
                        if "transfomer.layers.0" in par_name:  # sic!
                            if "weight" in par_name:
                                try:
                                    print(f"{par_name}, {param[0][0]}, {param[-1][-1]}")
                                except:
                                    continue
                        if i in (
                            0,
                            int(n_layers // 2),
                            n_layers - 4,
                            n_layers - 3,
                            n_layers - 2,
                            n_layers - 1,
                        ):
                            if "weight" in par_name:
                                print(
                                    f"{par_name}, {param[0][0]}, {param.grad[0][0]}, {param[-1][-1]}, {param.grad[-1][-1]}"
                                )

                evaluation_start = time()
                try:
                    final_result_dict, all_q_values = evaluate(
                        self.combined_model,
                        "",
                        self.device,
                        self.p_error_list,
                        self.p_msmt_list,
                        epsilon=self.learner_epsilon,
                        discount_factor_gamma=self.discount_factor,
                        num_of_random_episodes=120,
                        num_of_user_episodes=8,
                        verbosity=self.verbosity,
                        rl_type="ppo",
                    )

                except Exception as _:
                    error_traceback = traceback.format_exc()
                    self.logger.error("Caught exception in learning step")
                    self.logger.error(error_traceback)
                if self.benchmark:
                    evaluation_stop = time()
                    self.logger.info(
                        f"Time for evaluation: {evaluation_stop - evaluation_start} s."
                    )

                tb_results = {}
                for key, values in final_result_dict.items():
                    tb_results[key] = transform_list_dict(values)

                if self.verbosity:
                    log_evaluation_data(
                        self.tensorboard,
                        tb_results,
                        self.p_error_list,
                        t + 1,
                        current_time,
                    )

                    if self.verbosity >= 4:
                        for p_err in self.p_error_list:
                            self.tensorboard.add_histogram(
                                f"network/q_values, p_error {p_err}",
                                all_q_values[RESULT_KEY_HISTOGRAM_Q_VALUES],
                                t + 1,
                                walltime=current_time,
                            )

                eval_step += 1

                # monitor policy network parameters
                if self.verbosity >= 5:
                    policy_params = list(self.combined_model.parameters())
                    policy_named_params = list(self.combined_model.named_parameters())
                    n_layers = len(policy_params)
                    for i, param in enumerate(policy_params):
                        if i % 2 == 0:
                            layer_params = param.detach().cpu().numpy()
                            param_name = policy_named_params[i][0]
                            self.tensorboard.add_histogram(
                                f"learner/layer_{i}/{param_name}",
                                layer_params.reshape(-1, 1),
                                self.tensorboard_step,
                                walltime=current_time,
                            )

            self.tensorboard_step += 1

            if time() - self.heart > self.heartbeat_interval:
                self.heart = time()
                self.logger.debug(
                    "I'm alive my friend. I can see the shadows everywhere!"
                )

        self.logger.info("Reached maximum number of training steps. Terminate!")
        msg = ("terminate", None)
        for queue in self.worker_queues:
            queue[0].send(msg)

        save_model(self.combined_model, self.optimizer, None, self.save_model_path_date)
        self.logger.info(f"Saved policy network to {self.save_model_path_date}")

        self.tensorboard.close()
        exit()
