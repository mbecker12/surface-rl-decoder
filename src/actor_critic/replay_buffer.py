"""
This is the implementation of the replay buffer to be used in the PPO context.
Triggers the environment process and its underlying worker processes to
step through episodes and generate samples this way.
The samples are stored in arrays in this class.
"""
import logging
from time import time
import gc
import numpy as np

# pylint: disable=not-callable
import torch
from actor_critic.ppo_env import MultiprocessEnv
from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buffer")
logger.setLevel(logging.INFO)


class EpisodeBuffer:
    def __init__(self, args) -> None:
        self.num_workers = args["num_workers"]
        self.stack_depth = args["stack_depth"]
        self.code_size = args["code_size"]
        self.syndrome_size = self.code_size + 1
        self.gamma = args["discount_factor"]
        self.tau = args["episode_buffer_tau"]
        self.max_episodes = args["max_buffer_episodes"]
        self.max_steps = args["max_buffer_episode_steps"]
        self.device = args["episode_buffer_device"]

        self.total_received_samples = args["total_received_samples"]
        self.tensorboard = args["tensorboard"]
        self.verbosity = args["verbosity"]

        if self.verbosity >= 4:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.counter = 0

        assert self.max_episodes >= self.num_workers

        self.all_worker_idx = np.arange(self.num_workers)

        self._truncated_fn = np.vectorize(
            lambda x: "TimeLimit.truncated" in x and x["TimeLimit.truncated"]
        )

        self.discounts = np.logspace(
            0,
            self.max_steps + 1,
            num=self.max_steps + 1,
            base=self.gamma,
            endpoint=False,
            dtype=np.float128,
        )
        self.tau_discounts = np.logspace(
            0,
            self.max_steps + 1,
            num=self.max_steps + 1,
            base=self.tau * self.gamma,
            endpoint=False,
            dtype=np.float128,
        )

        self.clear()

    def clear(self):
        self.states_mem = np.empty(
            shape=np.concatenate(
                (
                    (self.max_episodes, self.max_steps),
                    (self.stack_depth, self.syndrome_size, self.syndrome_size),
                )
            ),
            dtype=np.uint8,
        )
        # self.states_mem[:] = np.nan # TODO why?

        self.actions_mem = np.empty(
            shape=(self.max_episodes, self.max_steps, 3), dtype=np.uint8
        )
        # self.actions_mem[:] = np.nan

        self.rewards_mem = np.empty(
            shape=(self.max_episodes, self.max_steps), dtype=np.float32
        )
        self.rewards_mem[:] = np.nan

        # gae = Generalized advantage estimation
        self.gaes_mem = np.empty(
            shape=(self.max_episodes, self.max_steps), dtype=np.float32
        )
        self.gaes_mem[:] = np.nan

        # log probability of action
        self.logpas_mem = np.empty(
            shape=(self.max_episodes, self.max_steps), dtype=np.float32
        )
        self.logpas_mem[:] = np.nan

        self.values_mem = np.empty(
            shape=(self.max_episodes, self.max_steps, 1), dtype=np.float32
        )
        self.values_mem[:] = np.nan

        self.episode_steps = np.zeros(shape=(self.max_episodes), dtype=np.uint16)
        self.episode_rewards = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_exploration = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_seconds = np.zeros(shape=(self.max_episodes), dtype=np.float64)

        self.current_ep_idxs = np.arange(self.num_workers, dtype=np.uint16)
        gc.collect()

    def fill(self, env_set: MultiprocessEnv, combined_model: BaseAgent):
        logger.debug("Filling episode buffer")
        states = env_set.reset()

        worker_rewards = np.zeros(
            shape=(self.num_workers, self.max_steps), dtype=np.float32
        )
        worker_exploratory = np.zeros(
            shape=(self.num_workers, self.max_steps), dtype=np.bool
        )
        worker_steps = np.zeros(shape=(self.num_workers), dtype=np.uint16)
        worker_seconds = np.array(
            [
                time(),
            ]
            * self.num_workers,
            dtype=np.float64,
        )

        buffer_full = False
        while (
            not buffer_full
            and len(self.episode_steps[self.episode_steps > 0])
            < self.max_episodes * 0.75
        ):
            with torch.no_grad():
                actions, logpas, are_exploratory, values = combined_model.np_pass(
                    states
                )

            next_states, rewards, terminals, infos = env_set.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.logpas_mem[self.current_ep_idxs, worker_steps] = logpas
            self.values_mem[self.current_ep_idxs, worker_steps] = values

            worker_exploratory[self.all_worker_idx, worker_steps] = are_exploratory
            worker_rewards[self.all_worker_idx, worker_steps] = rewards

            for w_idx in range(self.num_workers):
                if worker_steps[w_idx] + 1 == self.max_steps:
                    terminals[w_idx] = 1
                    infos[w_idx]["TimeLimit.truncated"] = True

            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
                next_values = np.zeros(shape=(self.num_workers, 1))
                truncated = self._truncated_fn(infos)
                if truncated.sum():
                    idx_truncated = np.flatnonzero(truncated)
                    with torch.no_grad():
                        _, tmp_next_values = combined_model(next_states[idx_truncated])
                        next_values[idx_truncated] = tmp_next_values.cpu().numpy()

            states = next_states
            worker_steps += 1

            if terminals.sum():
                new_states = env_set.reset(ranks=idx_terminals)
                states[idx_terminals] = new_states

                # for w_idx in range(self.num_workers):
                #     if w_idx not in idx_terminals:
                #         continue

                # should be equivalent to
                for w_idx in idx_terminals:
                    e_idx = self.current_ep_idxs[w_idx]
                    steps = worker_steps[w_idx]
                    self.episode_steps[e_idx] = steps
                    self.episode_rewards[e_idx] = worker_rewards[w_idx, :steps].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[
                        w_idx, :steps
                    ].mean()
                    self.episode_seconds[e_idx] = time() - worker_seconds[w_idx]

                    ep_rewards = np.concatenate(
                        (worker_rewards[w_idx, :steps], next_values[w_idx])
                    )
                    ep_discounts = self.discounts[: steps + 1]
                    ep_returns = np.array(
                        [
                            np.sum(ep_discounts[: steps + 1 - t] * ep_rewards[t:])
                            for t in range(steps)
                        ]
                    )
                    self.rewards_mem[e_idx, :steps] = ep_returns

                    ep_states = self.states_mem[e_idx, :steps]
                    with torch.no_grad():
                        _, values = combined_model(ep_states)
                        ep_values = torch.cat(
                            (
                                values,
                                torch.tensor(
                                    [next_values[w_idx]],
                                    device=combined_model.device,
                                    dtype=torch.float32,
                                ),
                            )
                        )
                        np_ep_values = ep_values.view(-1).cpu().numpy()
                        ep_tau_discounts = self.tau_discounts[:steps]
                        deltas = (
                            ep_rewards[:-1]
                            + self.gamma * np_ep_values[1:]
                            - np_ep_values[:-1]
                        )
                        gaes = np.array(
                            [
                                np.sum(self.tau_discounts[: steps - t] * deltas[t:])
                                for t in range(steps)
                            ]
                        )
                        self.gaes_mem[e_idx, :steps] = gaes

                        worker_exploratory[w_idx, :] = 0
                        worker_rewards[w_idx, :] = 0
                        worker_steps[w_idx] = 0
                        worker_seconds[w_idx] = time()

                        new_ep_id = max(self.current_ep_idxs) + 1
                        if new_ep_id >= self.max_episodes:
                            buffer_full = True
                            break

                        self.current_ep_idxs[w_idx] = new_ep_id

        # end while; buffer full or majority of episodes reached step limit

        ep_idxs = self.episode_steps > 0
        ep_steps = self.episode_steps[ep_idxs]

        # pylint: disable=attribute-defined-outside-init
        self.states_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.states_mem[ep_idxs])
        ]
        self.states_mem = np.concatenate(self.states_mem)

        self.actions_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.actions_mem[ep_idxs])
        ]
        self.actions_mem = np.concatenate(self.actions_mem)

        self.rewards_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.rewards_mem[ep_idxs])
        ]
        self.rewards_mem = torch.tensor(
            np.concatenate(self.rewards_mem), device=combined_model.device
        )

        self.gaes_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])
        ]
        self.gaes_mem = torch.tensor(
            np.concatenate(self.gaes_mem), device=combined_model.device
        )

        self.logpas_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.logpas_mem[ep_idxs])
        ]
        self.logpas_mem = torch.tensor(
            np.concatenate(self.logpas_mem), device=combined_model.device
        )

        self.values_mem = [
            row[: ep_steps[i]] for i, row in enumerate(self.values_mem[ep_idxs])
        ]
        self.values_mem = torch.tensor(
            np.concatenate(self.values_mem),
            device=combined_model.device,
            dtype=torch.float64,
        )

        ep_rewards = self.episode_rewards[ep_idxs]
        ep_exploration = self.episode_exploration[ep_idxs]
        ep_seconds = self.episode_seconds[ep_idxs]

        total_samples = len(self.states_mem)

        current_time = time()
        if self.verbosity:
            self.tensorboard.add_scalar(
                "io/buffer_recv_samples", total_samples, self.counter, current_time
            )
        self.counter += 1
        return ep_steps, ep_rewards, ep_exploration, ep_seconds

    def get_stacks(self):
        return (
            self.states_mem,
            self.actions_mem,
            self.rewards_mem,
            self.gaes_mem,
            self.logpas_mem,
            self.values_mem,
        )

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()
