import os
from time import time
import logging
import traceback
import nvgpu
import torch
import numpy as np
from torch.optim.adam import Adam
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
from evaluation.batch_evaluation import RESULT_KEY_HISTOGRAM_Q_VALUES
from evaluation.evaluate import evaluate
from distributed.learner_util import log_evaluation_data, transform_list_dict
from distributed.io_util import monitor_cpu_memory, monitor_gpu_memory

EPS = 1e-16
logging.basicConfig(level=logging.INFO)


class PPO:
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

        summary_path = env_args["summary_path"]
        summary_date = env_args["summary_date"]
        load_model_flag = learner_args["load_model"]
        old_model_path = learner_args["old_model_path"]
        save_model_path = learner_args["save_model_path"]
        model_name = learner_args["model_name"]
        model_config = learner_args["model_config"]

        self.save_model_path_date = os.path.join(
            save_model_path,
            str(self.code_size),
            summary_date,
            f"{model_name}_{self.code_size}_{summary_date}.pt",
        )

        self.model = None  # TODO choose model
        # TODO change model to one single model
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

        model_config = extend_model_config(
            model_config, self.syndrome_size, self.stack_depth, device=self.device
        )

        self.policy_model = choose_model(model_name, model_config)
        self.value_model = choose_model(model_name, model_config)

        model_config["rl_type"] = "ppo"
        self.combined_model = choose_model(model_name, model_config)

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
            self.policy_model.to(self.device)
            self.value_model.to(self.device)

            self.optimizer = Adam(
                self.combined_model.parameters(), lr=self.learning_rate
            )
            self.policy_optimizer = Adam(
                self.policy_model.parameters(), lr=self.learning_rate
            )
            self.value_optimizer = Adam(
                self.value_model.parameters(), lr=self.learning_rate
            )

        # initialize tensorboard
        self.tensorboard = SummaryWriter(
            os.path.join(summary_path, str(self.code_size), summary_date, "learner")
        )
        self.tensorboard_step = 0
        self.received_data = 0

        self.policy_model_max_grad_norm = learner_args.get("policy_model_max_grad_norm")
        self.policy_clip_range = learner_args.get("policy_clip_range")
        self.policy_stopping_kl = learner_args.get("policy_stopping_kl")
        self.value_model_max_grad_norm = learner_args.get("value_model_max_grad_norm")
        self.value_clip_range = learner_args.get("value_clip_range")
        self.value_stopping_mse = learner_args.get("value_stopping_mse")
        self.entropy_loss_weight = learner_args.get("entropy_loss_weight")
        self.value_loss_weight = learner_args.get("value_loss_weight")
        self.discount_factor = learner_args["discount_factor"]
        self.optimization_epochs = learner_args["optimization_epochs"]
        self.batch_size = learner_args["batch_size"]

        self.eval_frequency = learner_args["eval_frequency"]
        self.p_error_list = learner_args["learner_eval_p_error"]
        self.p_msmt_list = learner_args["learner_eval_p_msmt"]
        self.learner_epsilon = learner_args["learner_epsilon"]

        # self.tau = mem_args.get("tau")
        # episode_buffer_device = mem_args.get("episode_buffer_device")
        # self.max_buffer_episodes = mem_args.get("max_buffer_episodes")
        # self.max_buffer_episode_steps = mem_args.get("max_buffer_episode_steps")

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

        # TODO check best way for initialization
        self.logger.info("Set up multi process environment")
        self.worker_queues = queues["worker_queues"]
        self.env_set = MultiprocessEnv(env_args, worker_args, queues)

        self.total_learner_recv_samples = 0

    def optimize_model(self, current_timestep):
        (
            states,
            actions,
            returns,
            gaes,
            logpas,
            values,
        ) = self.episode_buffer.get_stacks()

        self.total_learner_recv_samples += states.shape[0]
        current_time = time()
        self.tensorboard.add_scalar(
            "io/learner_recv_samples",
            self.total_learner_recv_samples,
            current_timestep,
            current_time,
        )

        actions = torch.tensor(
            [action_to_q_value_index(action, self.code_size) for action in actions]
        )

        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)
        n_samples = len(actions)

        # TODO grokking loops over self.policy_optimization_epochs here
        # while True:
        self.logger.info("start optimization loop")
        for _ in range(self.optimization_epochs):
            # optimize policy model
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

            torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            (policy_loss + entropy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.combined_model.parameters(), self.policy_model_max_grad_norm
            )
            self.optimizer.step()

            # with torch.no_grad():
            #     # TODO: do we need this?
            #     logpas_pred_all, _, _ = self.combined_model.get_predictions_ppo(
            #         states, actions
            #     )
            #     kl = (logpas - logpas_pred_all).mean()
            #     if kl.item() > self.policy_stopping_kl:
            #         break

            # with torch.no_grad():
            #     _, values_pred_all = self.combined_model(states)
            #     mse = (values - values_pred_all).pow(2).mul(0.5).mean()
            #     if mse.item() > self.value_stopping_mse:
            #         break
        self.logger.info("end optimization loop")

    def train(self, seed):
        training_start, last_debug_time = time(), float("-inf")
        self.logger.info("Inside training function")

        # num_environments = args["num_environments"]
        # env = SurfaceCode()

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
        training_time = 0
        episode = 0
        performance_start = time()
        count_to_eval = 0
        eval_step = 0

        for t in range(self.max_timesteps):
            current_time = time()
            delta_t = current_time - performance_start

            if time() - self.start_time > self.max_time:
                self.logger.warning("Learner: time exceeded, aborting...")
                break

            # TODO: find a way to parallelize this
            (
                episode_timestep,
                episode_reward,
                episode_exploration,
                episode_seconds,
            ) = self.episode_buffer.fill(self.env_set, self.combined_model)

            if self.verbosity:
                if (
                    self.gpu_available
                    and self.nvidia_log_time > self.nvidia_log_frequency
                ):
                    monitor_gpu_memory(
                        self.tensorboard, current_time, performance_start, current_time
                    )
                if self.verbosity >= 3:
                    monitor_cpu_memory(
                        self.tensorboard, current_time, performance_start, current_time
                    )

            n_ep_batch = len(episode_timestep)
            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)

            self.logger.info("optimize model")
            try:
                self.optimize_model(t)
            except TypeError as _:
                error_traceback = traceback.format_exc()
                self.logger.error("Caught exception in learning step")
                self.logger.error(error_traceback)
            self.episode_buffer.clear()

            # stats
            # TODO evaluation
            if self.eval_frequency != -1 and count_to_eval >= self.eval_frequency:
                self.logger.info(f"Start Evaluation, Step {t+1}")
                count_to_eval = 0

                evaluation_start = time()
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
                    n_layers = len(policy_params)
                    for i, param in enumerate(policy_params):
                        if i == 0:
                            first_layer_params = param.detach().cpu().numpy()
                            self.tensorboard.add_histogram(
                                "learner/first_layer",
                                first_layer_params.reshape(-1, 1),
                                self.tensorboard_step,
                                walltime=current_time,
                            )

                        if i == n_layers - 2:
                            last_layer_params = param.detach().cpu().numpy()
                            self.tensorboard.add_histogram(
                                "learner/last_layer",
                                last_layer_params.reshape(-1, 1),
                                self.tensorboard_step,
                                walltime=current_time,
                            )

            self.tensorboard_step += 1
            count_to_eval += 1

            if time() - self.heart > self.heartbeat_interval:
                self.heart = time()
                self.logger.debug(
                    "I'm alive my friend. I can see the shadows everywhere!"
                )

        self.logger.info("Reach maximum number of training steps. Terminate!")
        msg = ("terminate", None)
        for queue in self.worker_queues:
            queue[0].send(msg)

        save_model(self.combined_model, self.optimizer, None, self.save_model_path_date)
        self.logger.info(f"Saved policy network to {self.save_model_path_date}")

        self.tensorboard.close()
