
import logging
import os
import time
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .azalea_agent import AzaleaAgent
from .monitor import monitor
from .parallel_player import Player
from .prep import torch_batch_replays
from .process_pool import ProcessPool
from .replay_buffer import ReplayBuffer
from .typing import SearchableEnv
from .utils import import_and_get


def train(policy, config, rundir, *,
          replaybuf: Optional[ReplayBuffer] = None) \
        -> str:
    """Train model."""

    os.makedirs(rundir, exist_ok=True)
    os.makedirs(f'{rundir}/checkpoints', exist_ok=True)
    monitor.init(f'{rundir}/log', log_steps=10)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    policy.seed(config['seed'])

    device = torch.device(config['device'])
    oversampling = config['replaybuf_oversampling']
    batch_size = config['batch_size']

    game_class = import_and_get(config['game'])
    game_factory = partial(game_class, board_size=config['board_size'])

    # initialize replay buffer with random policy
    pool = ProcessPool(num_workers=config['num_player_workers'])
    if replaybuf is None:
        replaybuf = initialize_replay_buffer(pool, game_factory,
                                             config['replaybuf_size'])

    loader = DataLoader(replaybuf,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=(device.type == 'cuda'),
                        num_workers=config['num_dataloader_workers'],
                        collate_fn=torch_batch_replays)
    optimizer = optim.SGD(policy.net.parameters(),
                          lr=config['lr_initial'],
                          momentum=config['momentum'],
                          weight_decay=config['l2_regularization'])
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_decay_epochs'],
        gamma=config['lr_decay'])

    policy.net.to(device)
    policy.net.train()
    policy.settings['move_exploration'] = True
    policy.settings['move_sampling'] = True

    # instantiate game and wrap it together with policy
    agent = AzaleaAgent(game_factory,
                        policy=policy,
                        device=config['device'])
    player = Player(pool, [agent])

    loss = 0
    step = 0
    start_time = time.time()
    for epoch in range(1, config['total_epochs'] + 1):
        scheduler.step()
        for batch in loader:
            monitor.step = step
            batch = game_class.random_reflect(batch)
            output, loss_ = supervised_step(policy.net, batch, train=True,
                                            optimizer=optimizer, device=device)
            loss += loss_

            # update replay buffer
            replaybuf.consume(batch_size / oversampling, player)

            monitor.add_scalar('optim/loss', loss_)
            monitor.add_scalar('optim/value_loss', output['value_loss'])
            monitor.add_scalar('optim/moves_loss', output['moves_loss'])
            monitor.add_scalar('optim/learning_rate',
                               optimizer.param_groups[0]['lr'])
            if monitor.step % 5000 == 0:
                for name, param in policy.net.named_parameters():
                    monitor.add_histogram('param/' + name, param)
                    monitor.add_histogram('grad/' + name, param.grad)

            if config['log_interval'] and step % config['log_interval'] == 0:
                sps = config['log_interval'] / (time.time() - start_time)
                monitor.add_scalar('steps_per_second', sps)
                loss /= config['log_interval']
                logging.info(f'step {step} loss {loss:.4f} steps/sec {sps:.2f}')
                loss = 0
                start_time = time.time()

            if (config['model_checkpoint_interval']
                    and step % config['model_checkpoint_interval'] == 0):
                save_checkpoint(policy, f'{rundir}/checkpoints/checkpoint.{step}',
                                optimizer=optimizer)

            step += 1

    player.stop()
    path = save_checkpoint(policy, f'{rundir}/checkpoints/final')
    monitor.close()
    return path


def supervised_step(model, batch, *,
                    train=False,
                    optimizer=None,
                    device='cpu'):
    """Process one batch for supervised or bandit tasks.
    """
    if train:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(train):
        if train:
            optimizer.zero_grad()
        for k in batch:
            batch[k] = batch[k].to(device)
        output, loss = model.run(batch, compute_loss=True)
        if train:
            loss.backward()
            optimizer.step()
    return output, loss.item()


def initialize_replay_buffer(pool: ProcessPool,
                             game_factory: Callable[[], SearchableEnv],
                             size: int) \
        -> ReplayBuffer:
    agent = AzaleaAgent(game_factory)
    player = Player(pool, [agent])
    examples, metrics = player.read(size)
    player.stop()
    buf = ReplayBuffer(examples)
    games = metrics['games']
    examples = len(buf)
    logging.info(f'replaybuf initialized with {games} games '
                 f'and {examples} examples')
    return buf


def save_checkpoint(policy, name, *,
                    optimizer=None,
                    replaybuf=None):
    """Save model (and replay buffer) checkpoint
    :param name: File base name
    :param replaybuf: Save also replay buffer
    """
    state = {
        'policy': policy.state_dict(),
    }
    if optimizer:
        state['optimizer'] = optimizer.state_dict()
    path = f'{name}.policy.pth'
    torch.save(state, path)
    logging.info(f'saved policy checkpoint to {path}')

    if replaybuf:
        rpath = f'{name}.replaybuf.pth'
        torch.save(replaybuf.state_dict(), rpath)
        logging.info(f'saved replay buffer checkpoint to {rpath}')
    return path
