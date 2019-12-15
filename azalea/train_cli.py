
import logging

import torch
import yaml
import click

from .policy import Policy
from .policy_trainer import train
from .replay_buffer import ReplayBuffer
from .version import __version__


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='YAML configuration file')
@click.option('--rundir', type=click.Path(), required=True,
              help='Directory to save results from training run')
@click.option('--startpos', type=click.Path(),
              help='YAML file of board starting positions')
@click.option('--model', type=click.Path(),
              help='Warm start training from model checkpoint')
@click.option('--replaybuf', type=click.Path(),
              help='Warm start training from replay buffer checkpoint')
def main(config, rundir, model, replaybuf, startpos):
    """Train a chess AI model.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'azalea {__version__}')

    config = yaml.load(open(config))
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if startpos:
        ss = yaml.load(open(startpos))
        config['start_positions'] = [s['fen'] for s in ss]

    if model:
        policy = Policy.load(model)
        logging.info(f'loaded model checkpoint from {model}')
    else:
        policy = Policy()
        policy.initialize(config)

    if replaybuf:
        replaybuf = ReplayBuffer.load(replaybuf)
        logging.info(f'loaded replay buffer checkpoint from {replaybuf}')

    train(policy, config, rundir,
          replaybuf=replaybuf)


if __name__ == '__main__':
    main()
