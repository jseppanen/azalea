
import os
import logging
import time
import json

import numpy as np
import torch

from .policy_trainer import PolicyTrainer
from .monitor import monitor


class Experiment:
    def __init__(self, rundir):
        os.makedirs(rundir, exist_ok=True)
        self.rundir = rundir
        monitor.init(f'{self.rundir}/log', log_steps=10)

    def restart(self, config):
        os.makedirs(f'{self.rundir}/checkpoints', exist_ok=True)
        with open(f'{self.rundir}/config.json', 'w') as f:
            json.dump(config, f)
        self.trainer = PolicyTrainer(config)
        self.train_iter = train(self.trainer, config, self.save_checkpoint)

    def close(self):
        self.trainer.close()
        monitor.close()

    def train(self, steps=None):
        if steps is None:
            steps = np.inf
        while steps:
            try:
                next(self.train_iter)
            except StopIteration:
                break
            steps -= 1
        return {'step': monitor.step + 1}

    def load_checkpoint(self, path):
        """Restore experiment checkpoint
        """
        state = torch.load(path)
        self.trainer.load_state_dict(state)
        logging.info(f'loaded model checkpoint from {path}')

    def load_replaybuf(self, path):
        """Restore experiment replaybuf
        """
        state = torch.load(path)
        self.trainer.load_replaybuf_state_dict(state)
        logging.info(f'loaded replay buffer checkpoint from {path}')

    def save_checkpoint(self, name, replaybuf=False):
        """Save experiment checkpoint
        :param name: File base name
        :param replaybuf: Save also replay buffer
        """
        path = f'{self.rundir}/{name}.policy.pth'
        torch.save(self.trainer.state_dict(), path)
        logging.info(f'saved policy checkpoint to {path}')
        if replaybuf:
            rpath = f'{self.rundir}/{name}.replaybuf.pth'
            torch.save(self.trainer.sampler.state_dict(), rpath)
            logging.info(f'saved replay buffer checkpoint to {rpath}')
        return path


def train(trainer, config, save):
    loss = 0
    t0 = time.time()
    for step in range(config['total_steps']):
        monitor.step = step
        metrics = trainer.step()
        loss += metrics['loss']

        if config['log_interval'] and step % config['log_interval'] == 0:
            sps = config['log_interval'] / (time.time() - t0)
            monitor.add_scalar('steps_per_second', sps)
            loss /= config['log_interval']
            logging.info(f'step {step} loss {loss:.4f} steps/sec {sps:.2f}')
            loss = 0
            t0 = time.time()

        if (config['model_checkpoint_interval']
                and step % config['model_checkpoint_interval'] == 0):
            save_replaybuf = (
                config['replaybuf_checkpoint_interval']
                and step % config['replaybuf_checkpoint_interval'] == 0)
            save(f'checkpoints/checkpoint.{step}', replaybuf=save_replaybuf)

        yield
