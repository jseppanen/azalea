# -*- encoding: utf-8 -*-

import logging

import numpy as np
import torch
import json
import click

import azalea as az
from azalea.game.hex import HexGame


@click.command()
@click.option('--rundir', type=click.Path(), required=True,
              help='Directory to save results from evaluation run')
@click.option('--models-list-file', type=click.Path(exists=True),
              help='List of model locations')
@click.option('--rounds', type=int, default=1,
              help='Number of evaluation rounds')
@click.option('--workers', type=int,
              help='Number of parallel workers')
@click.argument('models', nargs=-1, type=click.Path())
def main(rundir, models_list_file, rounds, workers, models):
    """Compare game AI models.
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    seed = 0xBAD5EED5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = list(models)
    if models_list_file:
        with open(models_list_file) as f:
            for m in f:
                models.append(m.strip())
    if len(models) < 2:
        raise click.UsageError('need at least two models to compare')

    scores, outcomes = compare(models, rounds, workers, device, seed)

    with open(f'{rundir}/results.json', 'w') as f:
        res = {
            'outcomes': list(outcomes.items()),
            'scores': scores,
            'models': models
        }
        json.dump(res, f)

    for i in np.argsort(scores)[::-1][:10]:
        print(f'{scores[i]:4.0f} {models[i]}')


def compare(models, eval_rounds, workers, device, seed):
    device = torch.device(device)
    agents = []
    for uri in models:
        logging.info(f'loading {uri}')
        p = az.AzaleaAgent(HexGame, path=uri, device=device)
        agents.append(p)

    # add random policy as reference point at 0 elo
    models = ['<random>'] + models
    agents = [az.AzaleaAgent(HexGame)] + agents

    for ag in agents:
        ag.seed(seed)
        seed = seed + 1
        # enable stochastic move sampling
        ag.settings['move_sampling'] = True

    outcomes = az.evaluate(agents, eval_rounds, num_workers=workers)
    scores = az.ranking.compute_ranking(len(agents), outcomes).tolist()

    # remove random policy
    scores = scores[1:]
    outcomes = dict(((i - 1, j - 1), outcomes[i, j])
                    for i, j in outcomes)
    return scores, outcomes


if __name__ == '__main__':
    main()
