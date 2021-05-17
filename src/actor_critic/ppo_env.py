"""
Implementation of environments for the PPO strategy.
Defines different communication routines and spawns multiple
worker processes.
"""
import logging
import multiprocessing as mp
from time import sleep
import numpy as np
import torch
from surface_rl_decoder.surface_code import SurfaceCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)

# pylint: disable=expression-not-assigned
class MultiprocessEnv:
    """
    Wrapper class to handle I/O with multiple workers.
    This spawns a new subprocess for each worker instance
    and provides different routines based for different
    messages.

    No instance of any neural network model is needed.
    Via interaction with e.g. the replay buffer class,
    this MultiprocessEnv class merely receives information
    about the suggested action and performs a step in the gym
    environment.
    """

    def __init__(self, env_args, worker_args, queues) -> None:
        self.num_cuda_workers = env_args["num_cuda_actors"]
        self.num_cpu_workers = env_args["num_cpu_actors"]
        self.num_workers = self.num_cpu_workers + self.num_cuda_workers

        self.worker_queues = queues["worker_queues"]

        self.worker_processes = []
        for i in range(self.num_workers):
            if i < self.num_cuda_workers:
                worker_args["device"] = "cuda"
            else:
                worker_args["device"] = "cpu"

            worker_args["worker_queue"] = self.worker_queues[i][1]
            worker_args["id"] = i

            self.worker_processes.append(
                mp.Process(target=self.worker, args=(worker_args,))
            )
            logger.info(f"Spawn worker process {i} on device {worker_args['device']}")
            self.worker_processes[i].start()
            sleep(1)

        self.terminals = {rank: False for rank in range(self.num_workers)}

    def worker(self, args):
        seed = int(args.get("seed", 0))
        worker_id = int(args.get("id"))
        worker_queue = args["worker_queue"]
        if seed != 0:
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
            torch.cuda.manual_seed(seed + worker_id)
            torch.cuda.manual_seed_all(seed + worker_id)

        logger.info(f"Initialize surface code, worker {worker_id}")
        env = SurfaceCode()

        while True:
            cmd, kwargs = worker_queue.recv()
            if cmd == "reset":
                worker_queue.send(env.reset(**kwargs))
            elif cmd == "step":
                worker_queue.send(env.step(**kwargs))
            elif cmd == "_past_limit":
                worker_queue.send(env.current_action_index >= env.max_actions)
            else:
                del env
                worker_queue.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.worker_queues[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [parent_end.send(msg) for parent_end, _ in self.worker_queues]

    def close(self, **kwargs):
        self.broadcast_msg(("close", kwargs))
        [w.join() for w in self.worker_processes]

    def reset(self, ranks=None, **kwargs):
        if not (ranks is None):
            [self.send_msg(("reset", {}), rank) for rank in ranks]
            return np.stack(
                [
                    parent_end.recv()
                    for rank, (parent_end, _) in enumerate(self.worker_queues)
                    if rank in ranks
                ]
            )

        # else: ranks is None => reset all
        self.broadcast_msg(("reset", kwargs))
        return np.stack([parent_end.recv() for parent_end, _ in self.worker_queues])

    def step(self, actions):
        assert len(actions) == self.num_workers, f"{len(actions)=}, {self.num_workers=}"

        [
            self.send_msg(("step", {"action": actions[rank]}), rank)
            for rank in range(self.num_workers)
        ]

        results = []

        for rank in range(self.num_workers):
            parent_end, _ = self.worker_queues[rank]
            new_state, reward, terminal, info = parent_end.recv()
            results.append((new_state, float(reward), float(terminal), info))

        # pylint: disable=not-an-iterable
        return [
            np.stack(block).squeeze() for block in np.array(results, dtype=object).T
        ]
