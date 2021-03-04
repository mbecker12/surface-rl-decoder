import os
from time import time, sleep
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logging
from distributed.dummy_agent import DummyModel
from distributed.learner_util import parameters_to_vector, vector_to_parameters, data_to_batch, predict_max_optimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learner")
logger.setLevel(logging.INFO)


def learner(args):
    learner_io_queue = args["learner_io_queue"]
    io_learner_queue = args["io_learner_queue"]
    verbosity = args["verbosity"]

    learning_rate = args["learning_rate"]
    device = args["device"]
    syndrome_size = args["syndrome_size"]
    stack_depth = args["stack_depth"]
    policy_update_steps = args["policy_update_steps"]
    discount_factor = args["discount_factor"]

    start_time = time()
    max_time_h = args["max_time"] # hours
    max_time = max_time_h * 60 * 60 # seconds

    heart = time()
    heartbeat_interval = 10  # seconds
    timesteps = 10000

    policy_net = DummyModel(syndrome_size, stack_depth)
    target_net = DummyModel(syndrome_size, stack_depth)
    policy_net.to(device)
    target_net.to(device)
    optimizer = Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="none")

    received_data = 0

    summary_path = args["summary_path"]
    summary_date = args["summary_date"]

    tensorboard = SummaryWriter(os.path.join(summary_path, summary_date, "learner"))
    tensorboard_step = 0
    for t in range(timesteps):
        if time() - start_time > max_time:
            logger.info("Learner: time exceeded, aborting...")
            break

        if t % policy_update_steps == 0 and t > 0:
            performance_stop = time()
            performance_start = time()

            params = parameters_to_vector(policy_net.parameters())
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device)


        if io_learner_queue.qsize == 0:
            logger.info("Learner waiting")

        data = io_learner_queue.get()
        if data is not None:
            logger.info(f"{len(data)=}")

            data_size = len(data)
            received_data += data_size

            if verbosity:
                tensorboard.add_scalar("learner/received_data", received_data, tensorboard_step)
                tensorboard_step += 1

        batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal, weights, indices = data_to_batch()

        policy_net.train()
        target_net.eval()

        # compute policy net output
        policy_output = policy_net(batch_state)
        policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

        # compute target network output
        # target_output = predictMax(target_net, batch_next_state, len(batch_next_state),grid_shift, system_size, device)
        target_output = predict_max_optimized(target_net, batch_next_state, syndrome_size, device)
        target_output = target_output.to(device)

        # compute loss and update replay memory
        y = batch_reward + ((~batch_terminal).type(torch.float32) * discount_factor * target_output)
        y = y.clamp(-100, 100)
        loss = criterion(y, policy_output)
        optimizer.zero_grad()

        loss = weights * loss
        
        # Compute priorities
        priorities = np.absolute(loss.cpu().detach().numpy())
        
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()

        # update priorities in replay_memory
        p_update = (indices, priorities)
        msg = ("priorities", p_update)
        learner_io_queue.put(msg)

        # TODO: evaluation here

        if time() - heart > heartbeat_interval:
            heart = time()
            logger.info("I'm alive my friend. I can see the shadows everywhere!")
            if verbosity > 1:
                tensorboard.add_scalar("learner/heartbeat", 1, 0)
                
        sleep(1)

    logger.info("Time's up. Terminate!")
    msg = ("terminate", None)
    learner_io_queue.put(msg)

    # TODO: save model
    tensorboard.close()
