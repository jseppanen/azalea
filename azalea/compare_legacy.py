# -*- encoding: utf-8 -*-

import logging

import numpy as np
import torch
import click

import azalea as az
from azalea.game.hex import HexGame


@click.command()
@click.argument('model', type=click.Path())
def main(model):
    """Compare game AI models.
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rounds = 100
    workers = 0  # problems with >0

    scores = compare(model, rounds, workers, device)

    for i in np.argsort(scores)[::-1][:10]:
        print(f'{scores[i]:4.0f}')


def compare(model, eval_rounds, workers, device):
    device = torch.device(device)
    agents = [
        az.LegacyAgent(path=model, device=device),
        az.AzaleaAgent(HexGame, path=model, device=device)
    ]

    for ag in agents:
        # enable stochastic move sampling
        ag.settings['move_sampling'] = True

    outcomes = az.evaluate(agents, eval_rounds, num_workers=workers)
    scores = az.ranking.compute_ranking(len(agents), outcomes).tolist()
    return scores


if __name__ == '__main__':
    main()
