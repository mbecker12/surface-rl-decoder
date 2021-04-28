from time import time
import logging
from numpy.core.fromnumeric import choose
import torch
import numpy as np
from torch.optim.adam import Adam
from actor_critic.ppo_env import MultiprocessEnv
from actor_critic.replay_buffer import EpisodeBuffer
from distributed.model_util import choose_model, extend_model_config
from distributed.mp_util import configure_processes
from distributed.util import action_to_q_value_index

EPS = 1e-16
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppo")
logger.setLevel(logging.INFO)


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

        self.model = None  # TODO choose model
        # TODO change model to one single model
        model_name = learner_args["model_name"]
        model_config = learner_args["model_config"]
        model_config = extend_model_config(
            model_config, self.syndrome_size, self.stack_depth, device=self.device
        )

        self.policy_model = choose_model(model_name, model_config)
        self.value_model = choose_model(model_name, model_config)

        model_config["rl_type"] = "ppo"
        self.combined_model = choose_model(model_name, model_config)

        self.combined_model.to(self.device)
        self.policy_model.to(self.device)
        self.value_model.to(self.device)

        self.optimizer = Adam(self.combined_model.parameters(), lr=self.learning_rate)
        self.policy_optimizer = Adam(
            self.policy_model.parameters(), lr=self.learning_rate
        )
        self.value_optimizer = Adam(
            self.value_model.parameters(), lr=self.learning_rate
        )

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

        logger.info("Initialize Episode Buffer")
        self.episode_buffer = EpisodeBuffer(mem_args)

        # TODO check best way for initialization
        logger.info("Set up multi process environment")
        self.env_set = MultiprocessEnv(env_args, worker_args, queues)

    def optimize_model(self):
        (
            states,
            actions,
            returns,
            gaes,
            logpas,
            values,
        ) = self.episode_buffer.get_stacks()

        actions = torch.tensor(
            [action_to_q_value_index(action, self.code_size) for action in actions]
        )

        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)
        n_samples = len(actions)

        # TODO grokking loops over self.policy_optimization_epochs here
        # while True:
        logger.info("start optimization loop")
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

            # # optimize value separately ?
            # _, _, values_pred = self.combined_model.get_predictions_ppo(states_batch, actions_batch)

            # self.optimizer.zero_grad()
            # value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.value_model_max_grad_norm)

            # self.optimizer.step()

            with torch.no_grad():
                # TODO: do we need this?
                logpas_pred_all, _, _ = self.combined_model.get_predictions_ppo(
                    states, actions
                )
                kl = (logpas - logpas_pred_all).mean()
                if kl.item() > self.policy_stopping_kl:
                    break

            with torch.no_grad():
                _, values_pred_all = self.combined_model(states)
                mse = (values - values_pred_all).pow(2).mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    break

    def train(self, seed):
        training_start, last_debug_time = time(), float("-inf")
        logger.info("Inside training function")

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

        while True:
            (
                episode_timestep,
                episode_reward,
                episode_exploration,
                episode_seconds,
            ) = self.episode_buffer.fill(self.env_set, self.combined_model)

            n_ep_batch = len(episode_timestep)
            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)

            logger.info("optimize model")
            self.optimize_model()
            self.episode_buffer.clear()

            # stats
            # TODO evaluation

            # TODO finish training after certain time

            # TODO save model

            # TODO tensorboard support
